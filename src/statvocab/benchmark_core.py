"""Core-corpus performance benchmark reporting."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from statvocab.config import AppConfig


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_json(pattern: str) -> dict[str, Any]:
    candidates = sorted(Path().glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return {}
    return _read_json(candidates[0])


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _rss_bytes() -> int | None:
    try:
        psutil = import_module("psutil")
        return int(psutil.Process().memory_info().rss)
    except ModuleNotFoundError:
        return None


def _stage(
    *,
    name: str,
    wall_clock_seconds: float | None = None,
    peak_rss_bytes: int | None = None,
    throughput: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage": name,
        "wall_clock_seconds": wall_clock_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "throughput": throughput or {},
        "diagnostics": diagnostics or {},
    }


def write_core_benchmark(config: AppConfig) -> tuple[Path, dict[str, Any]]:
    """Write a benchmark snapshot over the current 2,000-table core artifacts."""

    if config.corpus.name != "core":
        raise ValueError("benchmark-core must use the core corpus config.")

    ingest = _read_json(config.paths.processed_dir / "ingestion_diagnostics.json")
    extraction = _read_json(config.paths.processed_dir / "extraction_diagnostics.json")
    classification = _latest_json("outputs/classification/run_*/classification_summary.json")
    clustering = _latest_json("outputs/clustering/run_*/clustering_summary.json")
    relations = _read_json(config.paths.reports_dir / "relations_metrics.json").get(
        "latest_run_summary",
        {},
    )
    search = _read_json(config.paths.reports_dir / "retrieval_metrics.json").get("index", {})
    retrieval = _read_json(config.paths.reports_dir / "retrieval_metrics.json")

    table_count = int(
        extraction.get("completed_table_count") or ingest.get("inventoried_table_count") or 0
    )
    vocabulary_count = int(extraction.get("vocabulary_count") or 0)
    relation_candidates = int(relations.get("candidate_count") or 0)

    stages = {
        "ingest": _stage(
            name="ingest",
            wall_clock_seconds=ingest.get("wall_clock_seconds"),
            throughput={
                "tables_per_second": (
                    int(ingest.get("inventoried_table_count") or 0)
                    / float(ingest["wall_clock_seconds"])
                    if ingest.get("wall_clock_seconds")
                    else None
                ),
            },
            diagnostics={
                "inventoried_table_count": ingest.get("inventoried_table_count"),
                "status_counts": ingest.get("status_counts", {}),
            },
        ),
        "extraction": _stage(
            name="extraction",
            wall_clock_seconds=extraction.get("wall_clock_seconds"),
            throughput={
                "tables_per_minute": extraction.get("tables_per_minute"),
                "terms_per_second": (
                    vocabulary_count / float(extraction["wall_clock_seconds"])
                    if extraction.get("wall_clock_seconds")
                    else None
                ),
            },
            diagnostics={
                "table_count": table_count,
                "vocabulary_count": vocabulary_count,
                "parallel_workers": extraction.get("parallel_workers"),
                "parallel_chunk_size": extraction.get("parallel_chunk_size"),
            },
        ),
        "classification": _stage(
            name="classification",
            wall_clock_seconds=sum(
                float(value)
                for value in (classification.get("timing") or {}).values()
                if isinstance(value, int | float)
            )
            or None,
            throughput={
                "terms_per_second": (
                    int(classification.get("prediction_count") or 0)
                    / sum(
                        float(value)
                        for value in (classification.get("timing") or {}).values()
                        if isinstance(value, int | float)
                    )
                    if classification.get("timing")
                    else None
                ),
            },
            diagnostics={
                "run_id": classification.get("run_id"),
                "variant": classification.get("variant"),
                "prediction_count": classification.get("prediction_count"),
                "timing": classification.get("timing") or {},
            },
        ),
        "clustering": _stage(
            name="clustering",
            wall_clock_seconds=clustering.get("wall_clock_seconds"),
            diagnostics={
                "run_id": clustering.get("run_id"),
                "measure_count": clustering.get("measure_count"),
                "coverage": clustering.get("coverage"),
                "contextual_features_enabled": clustering.get("contextual_features_enabled"),
                "timing": clustering.get("timing") or {},
            },
        ),
        "relations": _stage(
            name="relations",
            wall_clock_seconds=relations.get("wall_clock_seconds"),
            throughput={
                "relation_candidates_per_second": relations.get(
                    "relation_candidates_per_second"
                ),
                "candidates_per_second": (
                    relation_candidates / float(relations["wall_clock_seconds"])
                    if relations.get("wall_clock_seconds")
                    else None
                ),
            },
            diagnostics={
                "candidate_count": relations.get("candidate_count"),
                "accepted_count": relations.get("accepted_count"),
                "all_pair_count": relations.get("all_pair_count"),
                "topk_pair_count": relations.get("topk_pair_count"),
                "semantic_neighbor_k": relations.get("semantic_neighbor_k"),
            },
        ),
        "search_index": _stage(
            name="search_index",
            wall_clock_seconds=search.get("build_seconds"),
            peak_rss_bytes=search.get("rss_bytes"),
            throughput={
                "documents_per_second": (
                    int(search.get("document_count") or 0) / float(search["build_seconds"])
                    if search.get("build_seconds")
                    else None
                ),
            },
            diagnostics={
                "run_id": search.get("run_id"),
                "document_count": search.get("document_count"),
                "total_index_size_bytes": search.get("total_index_size_bytes"),
            },
        ),
        "retrieval": _stage(
            name="retrieval",
            diagnostics={
                "run_id": retrieval.get("run_id"),
                "selected_preset": (retrieval.get("tuning") or {}).get("selected_preset"),
                "systems": retrieval.get("systems", {}),
            },
        ),
    }
    artifacts = {
        "vocabulary": _file_info(config.paths.processed_dir / "vocabulary.parquet"),
        "term_embeddings": _file_info(config.paths.processed_dir / "term_embeddings.parquet"),
        "measures": _file_info(config.paths.outputs_dir / "measures.csv"),
        "measure_clusters": _file_info(config.paths.outputs_dir / "measure_clusters.csv"),
        "measure_relations": _file_info(config.paths.outputs_dir / "measure_relations.csv"),
        "current_index": _file_info(config.paths.outputs_dir / "search" / "current_index.json"),
    }
    payload = {
        "config_name": config.config_name,
        "corpus": config.corpus.name,
        "expected_table_count": config.corpus.expected_table_count,
        "peak_rss_bytes": _rss_bytes(),
        "selected_config": {
            "extraction_parallel_workers": config.extraction.parallel_workers,
            "extraction_parallel_chunk_size": config.extraction.parallel_chunk_size,
            "relations_semantic_neighbor_k": config.relations.semantic_neighbor_k,
            "relations_max_candidates_per_measure": config.relations.max_candidates_per_measure,
            "classification_embedding_batch_size": config.classification.embedding_batch_size,
            "search_semantic_top_k": config.search.semantic_top_k,
            "search_lexical_top_k": config.search.lexical_top_k,
        },
        "stages": stages,
        "artifact_sizes": artifacts,
    }
    path = config.paths.reports_dir / "performance_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, payload
