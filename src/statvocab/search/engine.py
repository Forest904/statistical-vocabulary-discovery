"""Search index build orchestration and query-time engine."""

from __future__ import annotations

import json
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from statvocab.classification_features import read_parquet_rows
from statvocab.config import AppConfig
from statvocab.contracts import ResourceRecord
from statvocab.manifests import complete_manifest, create_manifest, write_manifest
from statvocab.resources import file_md5
from statvocab.search.documents import build_search_documents
from statvocab.search.explain import NOTICE, explain_candidate
from statvocab.search.lexical import LexicalHit, build_lexical_index, search_lexical
from statvocab.search.query_parser import ParsedQuery, parse_query
from statvocab.search.rank import RankedCandidate, _normalize_scores, rank_candidates
from statvocab.search.semantic import (
    PearlEmbedder,
    SemanticHit,
    build_semantic_index,
    search_semantic,
)

SearchSystem = Literal["fused", "title_bm25", "all_vocabulary_bm25", "pearl_semantic"]
__all__ = ["NOTICE", "SearchEngine", "build_search_index"]


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "size_bytes": 0, "md5": None}
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "md5": file_md5(path),
    }


def _rss_bytes() -> int | None:
    try:
        psutil = import_module("psutil")
        current_process = psutil.Process()
        return int(current_process.memory_info().rss)
    except ModuleNotFoundError:
        return None


def build_search_index(
    config: AppConfig,
    *,
    resources: tuple[ResourceRecord, ...] = (),
) -> tuple[dict[str, Path], Path, dict[str, Any]]:
    """Build table search documents, SQLite lexical index, and FAISS semantic index."""

    manifest = create_manifest(config, "build-search-index")
    started = time.perf_counter()
    output_dir = config.paths.outputs_dir / "search" / manifest.run_id
    rows, documents_path = build_search_documents(config, run_id=manifest.run_id)
    lexical_path = build_lexical_index(rows, output_dir / "lexical.sqlite")
    semantic_path, semantic_metadata_path, dimension = build_semantic_index(
        config,
        rows,
        index_path=output_dir / "semantic.faiss",
        metadata_path=output_dir / "semantic_metadata.parquet",
    )
    build_seconds = time.perf_counter() - started
    artifacts = {
        "search_documents": documents_path,
        "lexical_index": lexical_path,
        "semantic_index": semantic_path,
        "semantic_metadata": semantic_metadata_path,
    }
    files = {
        "search_documents": _file_info(documents_path),
        "lexical_index": _file_info(lexical_path),
        "semantic_index": _file_info(semantic_path),
        "semantic_metadata": _file_info(semantic_metadata_path),
    }
    summary = {
        "run_id": manifest.run_id,
        "corpus": config.corpus.name,
        "document_count": len(rows),
        "pearl_model": config.classification.pearl_model,
        "pearl_revision": config.classification.pearl_revision,
        "embedding_dimension": dimension,
        "build_seconds": build_seconds,
        "rss_bytes": _rss_bytes(),
        "files": files,
    }
    total_size = sum(int(item["size_bytes"]) for item in files.values())
    summary["total_index_size_bytes"] = total_size
    summary_path = _write_json(summary, output_dir / "search_index_summary.json")
    index_manifest_path = _write_json(
        {
            "run_id": manifest.run_id,
            "documents": str(documents_path),
            "lexical_index": str(lexical_path),
            "semantic_index": str(semantic_path),
            "semantic_metadata": str(semantic_metadata_path),
            "summary": str(summary_path),
        },
        output_dir / "index_manifest.json",
    )
    current_path = _write_json(
        {
            "run_id": manifest.run_id,
            "index_manifest": str(index_manifest_path),
            "lexical_index": str(lexical_path),
            "semantic_index": str(semantic_path),
            "semantic_metadata": str(semantic_metadata_path),
            "search_documents": str(documents_path),
        },
        config.paths.outputs_dir / "search" / "current_index.json",
    )
    artifacts |= {
        "summary": summary_path,
        "index_manifest": index_manifest_path,
        "current_index": current_path,
    }
    completed = complete_manifest(
        manifest,
        resources=tuple(record.resource_id for record in resources),
        artifacts=tuple(str(path) for path in artifacts.values()),
    )
    manifest_path = write_manifest(
        completed,
        config.paths.outputs_dir / "manifests" / f"{manifest.run_id}_build_search_index.json",
    )
    return artifacts, manifest_path, summary


class SearchEngine:
    """Loaded search engine over one completed index run."""

    def __init__(self, config: AppConfig, index_manifest_path: Path | None = None) -> None:
        self.config = config
        self.index_manifest_path = index_manifest_path or self._current_manifest_path()
        manifest = json.loads(self.index_manifest_path.read_text(encoding="utf-8"))
        self.run_id = str(manifest["run_id"])
        self.lexical_path = Path(str(manifest["lexical_index"]))
        self.semantic_path = Path(str(manifest["semantic_index"]))
        self.semantic_metadata_path = Path(str(manifest["semantic_metadata"]))
        self.documents_path = Path(str(manifest["documents"]))
        document_rows = read_parquet_rows(self.documents_path)
        self.documents = {str(row["table_id"]): row for row in document_rows}
        self.embedder = PearlEmbedder(config)

    def _current_manifest_path(self) -> Path:
        current_path = self.config.paths.outputs_dir / "search" / "current_index.json"
        if not current_path.exists():
            raise FileNotFoundError(
                f"Missing {current_path}; run `statvocab build-search-index` first."
            )
        current = json.loads(current_path.read_text(encoding="utf-8"))
        return Path(str(current["index_manifest"]))

    def _baseline_candidates(
        self,
        query: ParsedQuery,
        hits: list[LexicalHit] | list[SemanticHit],
        *,
        component: str,
        limit: int,
    ) -> list[RankedCandidate]:
        raw = {
            hit.table_id: max(0.0, hit.score)
            for hit in hits
            if hit.table_id in self.documents
        }
        normalized = _normalize_scores(raw)
        ranked = [
            RankedCandidate(
                table_id=table_id,
                score=score,
                score_components={
                    "semantic": score if component == "semantic" else 0.0,
                    "lexical": score if component == "lexical" else 0.0,
                    "measure": 0.0,
                    "dimension": 0.0,
                    "geography": 0.0,
                    "time": 0.0,
                },
                document=self.documents[table_id],
            )
            for table_id, score in normalized.items()
            if table_id in self.documents and score >= self.config.search.min_fused_score
        ]
        _ = query
        ranked.sort(key=lambda item: (-item.score, item.table_id))
        return ranked[:limit]

    def rank(
        self,
        query_text: str,
        *,
        limit: int | None = None,
        system: SearchSystem = "fused",
    ) -> tuple[ParsedQuery, list[RankedCandidate]]:
        """Return parsed query and ranked candidates for the requested system."""

        parsed = parse_query(self.config, query_text)
        actual_limit = limit or self.config.search.default_limit
        if parsed.is_blank:
            return parsed, []
        if system == "title_bm25":
            lexical_hits = search_lexical(
                self.lexical_path,
                parsed.original,
                top_k=max(actual_limit, self.config.search.lexical_top_k),
                mode="title",
            )
            return parsed, self._baseline_candidates(
                parsed,
                lexical_hits,
                component="lexical",
                limit=actual_limit,
            )
        if system == "all_vocabulary_bm25":
            lexical_hits = search_lexical(
                self.lexical_path,
                parsed.original,
                top_k=max(actual_limit, self.config.search.lexical_top_k),
                mode="all",
            )
            return parsed, self._baseline_candidates(
                parsed,
                lexical_hits,
                component="lexical",
                limit=actual_limit,
            )
        if system == "pearl_semantic":
            semantic_hits = search_semantic(
                self.config,
                index_path=self.semantic_path,
                metadata_path=self.semantic_metadata_path,
                query=parsed.semantic_remainder,
                top_k=max(actual_limit, self.config.search.semantic_top_k),
                embedder=self.embedder,
            )
            return parsed, self._baseline_candidates(
                parsed,
                semantic_hits,
                component="semantic",
                limit=actual_limit,
            )

        lexical_hits = search_lexical(
            self.lexical_path,
            parsed.original,
            top_k=self.config.search.lexical_top_k,
            mode="all",
        )
        semantic_hits = search_semantic(
            self.config,
            index_path=self.semantic_path,
            metadata_path=self.semantic_metadata_path,
            query=parsed.semantic_remainder,
            top_k=self.config.search.semantic_top_k,
            embedder=self.embedder,
        )
        return parsed, rank_candidates(
            query=parsed,
            documents=self.documents,
            lexical_hits=lexical_hits,
            semantic_hits=semantic_hits,
            config=self.config.search,
            limit=actual_limit,
        )

    def search(
        self,
        query_text: str,
        *,
        limit: int | None = None,
        system: SearchSystem = "fused",
    ) -> dict[str, Any]:
        """Return a grounded search response."""

        parsed, ranked = self.rank(query_text, limit=limit, system=system)
        return {
            "query": {
                "original": parsed.original,
                "semantic_remainder": parsed.semantic_remainder,
                "geographies": [
                    {
                        "code": item.code,
                        "name": item.name,
                        "raw_value": item.raw_value,
                        "normalized_value": item.normalized_value,
                    }
                    for item in parsed.geographies
                ],
                "times": [
                    {
                        "raw_value": item.raw_value,
                        "normalized_value": item.normalized_value,
                        "start_date": item.start_date,
                        "end_date": item.end_date,
                        "granularity": item.granularity,
                    }
                    for item in parsed.times
                ],
            },
            "notice": NOTICE,
            "results": [
                explain_candidate(rank=index, query=parsed, candidate=candidate)
                for index, candidate in enumerate(ranked, start=1)
            ],
        }
