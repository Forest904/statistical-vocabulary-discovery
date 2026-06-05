"""Read-only artifact loading and indexing for the FastAPI backend."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.app.settings import ApiSettings
from statvocab.classification import CSV_BY_CATEGORY
from statvocab.classification_features import read_parquet_rows
from statvocab.config import AppConfig, load_config
from statvocab.contracts import RelationType
from statvocab.resources import file_md5
from statvocab.search.engine import SearchEngine
from statvocab.search.explain import NOTICE

REQUIRED_PROCESSED_PARQUET = (
    "tables.parquet",
    "vocabulary.parquet",
    "table_vocabulary.parquet",
)
REQUIRED_OUTPUT_CSV = (
    *tuple(CSV_BY_CATEGORY.values()),
    "measure_clusters.csv",
    "measure_relations.csv",
)


class ApiStartupError(RuntimeError):
    """Raised when the API cannot load a usable artifact set."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.code = code
        self.details = details or {}
        super().__init__(message)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    _raise_csv_field_limit()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _read_required_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ApiStartupError(
            "missing_artifact",
            f"Missing required CSV artifact: {path}",
            {"path": str(path)},
        )
    rows = _read_csv(path)
    if not rows:
        raise ApiStartupError(
            "empty_artifact",
            f"Required CSV artifact is empty: {path}",
            {"path": str(path)},
        )
    return rows


def _read_required_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        rows = read_parquet_rows(path)
    except FileNotFoundError as exc:
        raise ApiStartupError(
            "missing_artifact",
            str(exc),
            {"path": str(path)},
        ) from exc
    if not rows:
        raise ApiStartupError(
            "empty_artifact",
            f"Required Parquet artifact is empty: {path}",
            {"path": str(path)},
        )
    return rows


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ApiStartupError(
            "missing_artifact",
            f"Missing required JSON artifact: {path}",
            {"path": str(path)},
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _json_list(value: object) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _json_dict(value: object) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _page(
    items: list[dict[str, Any]],
    *,
    page: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    offset = (page - 1) * page_size
    return items[offset : offset + page_size], len(items)


def _artifact_path(value: object) -> Path:
    raw = str(value)
    path = Path(raw)
    if path.exists():
        return path
    return Path(raw.replace("\\", "/"))


def _latest_json(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def validate_search_artifacts(config: AppConfig) -> dict[str, Any]:
    """Validate current index manifest paths and checksum metadata."""

    current_path = config.paths.outputs_dir / "search" / "current_index.json"
    if not current_path.exists():
        raise ApiStartupError(
            "missing_current_index",
            (
                f"Missing {current_path}; run `statvocab build-search-index "
                f"--config {config.config_name}` first."
            ),
            {"path": str(current_path)},
        )

    current = _read_json(current_path)
    manifest_path = _artifact_path(current.get("index_manifest", ""))
    if not manifest_path.exists():
        raise ApiStartupError(
            "missing_index_manifest",
            "The current search index points to a missing index manifest.",
            {"current_index": str(current_path), "index_manifest": str(manifest_path)},
        )

    manifest = _read_json(manifest_path)
    required_keys = ("documents", "lexical_index", "semantic_index", "semantic_metadata", "summary")
    resolved: dict[str, Path] = {"index_manifest": manifest_path}
    for key in required_keys:
        path = _artifact_path(manifest.get(key, ""))
        if not path.exists():
            raise ApiStartupError(
                "missing_index_file",
                f"The search index manifest references a missing {key} file.",
                {"key": key, "path": str(path), "index_manifest": str(manifest_path)},
            )
        resolved[key] = path

    summary = _read_json(resolved["summary"])
    files = _json_dict(summary.get("files"))
    for key in ("search_documents", "lexical_index", "semantic_index", "semantic_metadata"):
        artifact_key = "documents" if key == "search_documents" else key
        path = resolved[artifact_key]
        expected = _json_dict(files.get(key))
        if not expected:
            raise ApiStartupError(
                "missing_index_checksum",
                f"The search index summary does not include checksum metadata for {key}.",
                {"summary": str(resolved["summary"]), "key": key},
            )
        actual_size = path.stat().st_size
        expected_size = int(expected.get("size_bytes") or -1)
        expected_md5 = expected.get("md5")
        actual_md5 = file_md5(path)
        if actual_size != expected_size or actual_md5 != expected_md5:
            raise ApiStartupError(
                "stale_index_file",
                f"The {key} artifact does not match the recorded search index summary.",
                {
                    "path": str(path),
                    "expected_size_bytes": expected_size,
                    "actual_size_bytes": actual_size,
                    "expected_md5": expected_md5,
                    "actual_md5": actual_md5,
                },
            )

    return {
        "current": current,
        "manifest": manifest,
        "summary": summary,
        "paths": {key: str(path) for key, path in resolved.items()},
    }


def validate_required_artifact_paths(config: AppConfig) -> None:
    """Fail early when Milestone 7's read-only artifact surface is incomplete."""

    for filename in REQUIRED_PROCESSED_PARQUET:
        path = config.paths.processed_dir / filename
        if not path.exists():
            raise ApiStartupError(
                "missing_artifact",
                f"Missing required Parquet artifact: {path}",
                {"path": str(path)},
            )
    for filename in REQUIRED_OUTPUT_CSV:
        path = config.paths.outputs_dir / filename
        if not path.exists():
            raise ApiStartupError(
                "missing_artifact",
                f"Missing required CSV artifact: {path}",
                {"path": str(path)},
            )


def validate_loaded_artifacts(
    *,
    tables: dict[str, dict[str, Any]],
    search_documents: dict[str, dict[str, Any]],
    terms: dict[str, dict[str, Any]],
    term_outputs: dict[str, dict[str, Any]],
    table_terms: dict[str, list[dict[str, Any]]],
    clusters: list[dict[str, Any]],
    cluster_by_term: dict[str, dict[str, Any]],
    relations: list[dict[str, Any]],
) -> None:
    """Validate cross-artifact references after all rows are loaded."""

    table_ids = set(tables)
    document_ids = set(search_documents)
    term_ids = set(terms)
    output_term_ids = set(term_outputs)

    missing_documents = sorted(table_ids - document_ids)
    extra_documents = sorted(document_ids - table_ids)
    if missing_documents or extra_documents:
        raise ApiStartupError(
            "artifact_mismatch",
            "Search documents and table inventory do not reference the same table IDs.",
            {
                "missing_search_documents": missing_documents[:20],
                "extra_search_documents": extra_documents[:20],
                "missing_count": len(missing_documents),
                "extra_count": len(extra_documents),
            },
        )

    uncategorized_terms = sorted(term_ids - output_term_ids)
    unknown_output_terms = sorted(output_term_ids - term_ids)
    if uncategorized_terms or unknown_output_terms:
        raise ApiStartupError(
            "artifact_mismatch",
            "Final category CSVs do not match the vocabulary artifact.",
            {
                "uncategorized_terms": uncategorized_terms[:20],
                "unknown_output_terms": unknown_output_terms[:20],
                "uncategorized_count": len(uncategorized_terms),
                "unknown_output_count": len(unknown_output_terms),
            },
        )

    unknown_table_terms = sorted(
        table_id
        for table_id in table_terms
        if table_id not in table_ids
    )
    if unknown_table_terms:
        raise ApiStartupError(
            "artifact_mismatch",
            "Table vocabulary references unknown table IDs.",
            {
                "unknown_table_ids": unknown_table_terms[:20],
                "unknown_count": len(unknown_table_terms),
            },
        )

    unknown_cluster_terms = sorted(set(cluster_by_term) - term_ids)
    if unknown_cluster_terms:
        raise ApiStartupError(
            "artifact_mismatch",
            "Measure clusters reference unknown term IDs.",
            {
                "unknown_term_ids": unknown_cluster_terms[:20],
                "unknown_count": len(unknown_cluster_terms),
            },
        )
    if not clusters:
        raise ApiStartupError(
            "empty_artifact",
            "No measure clusters were loaded from measure_clusters.csv.",
            {"path": "measure_clusters.csv"},
        )

    allowed_relation_types = {relation_type.value for relation_type in RelationType}
    invalid_relations = [
        relation
        for relation in relations
        if relation.get("relation_type") not in allowed_relation_types
        or relation.get("source_term_id") not in term_ids
        or relation.get("target_term_id") not in term_ids
    ]
    if invalid_relations:
        raise ApiStartupError(
            "artifact_mismatch",
            "Measure relations contain invalid types or unknown term references.",
            {"invalid_relations": invalid_relations[:20], "invalid_count": len(invalid_relations)},
        )


@dataclass(frozen=True)
class ApiState:
    """Loaded read-only API state."""

    config: AppConfig
    search_engine: SearchEngine
    search_validation: dict[str, Any]
    loaded_at: datetime
    warnings: list[str]
    tables: dict[str, dict[str, Any]]
    search_documents: dict[str, dict[str, Any]]
    terms: dict[str, dict[str, Any]]
    term_outputs: dict[str, dict[str, Any]]
    table_terms: dict[str, list[dict[str, Any]]]
    term_tables: dict[str, list[dict[str, Any]]]
    clusters: list[dict[str, Any]]
    cluster_by_term: dict[str, dict[str, Any]]
    relations: list[dict[str, Any]]
    relations_by_term: dict[str, list[dict[str, Any]]]
    evaluation: dict[str, dict[str, Any]]

    @property
    def loaded_search_run_id(self) -> str:
        return self.search_engine.run_id

    @property
    def artifact_counts(self) -> dict[str, int]:
        return {
            "tables": len(self.tables),
            "search_documents": len(self.search_documents),
            "terms": len(self.terms),
            "clusters": len(self.clusters),
            "relations": len(self.relations),
        }

    def readiness(self) -> dict[str, bool]:
        return {
            "config_loaded": True,
            "search_index_ready": True,
            "tables_loaded": bool(self.tables),
            "terms_loaded": bool(self.terms),
            "clusters_loaded": bool(self.clusters),
            "relations_loaded": bool(self.relations),
            "evaluation_loaded": bool(self.evaluation),
        }

    def health(self) -> dict[str, Any]:
        readiness = self.readiness()
        return {
            "status": "ok" if all(readiness.values()) else "degraded",
            "config_name": self.config.config_name,
            "corpus": self.config.corpus.name,
            "loaded_search_run_id": self.loaded_search_run_id,
            "document_count": len(self.search_documents),
            "readiness": readiness,
            "artifact_counts": self.artifact_counts,
            "warnings": self.warnings,
        }

    def search(self, query: str, *, page: int, page_size: int, system: str) -> dict[str, Any]:
        requested = page * page_size
        payload = self.search_engine.search(query, limit=requested, system=system)  # type: ignore[arg-type]
        items, total = _page(list(payload["results"]), page=page, page_size=page_size)
        return {
            "query": payload["query"],
            "notice": payload.get("notice") or NOTICE,
            "pagination": {"page": page, "page_size": page_size, "total": total},
            "results": items,
        }

    def table_detail(self, table_id: str) -> dict[str, Any] | None:
        table = self.tables.get(table_id)
        document = self.search_documents.get(table_id)
        if table is None or document is None:
            return None
        related_terms = self.table_terms.get(table_id, [])
        return {
            "table_id": table_id,
            "title": str(document.get("title_clean") or table.get("title") or ""),
            "source_url": str(document.get("source_url") or table.get("source_url") or ""),
            "parse_status": str(table.get("parse_status") or document.get("parse_status") or ""),
            "metadata": {
                "filename": table.get("filename"),
                "file_md5": table.get("file_md5"),
                "row_count": table.get("row_count"),
                "observation_count": table.get("observation_count"),
                "malformed_row_count": table.get("malformed_row_count"),
                "warning_count": table.get("warning_count"),
                "warning_reasons": table.get("warning_reasons") or [],
                "source_repaired": table.get("source_repaired"),
                "repair_reason": table.get("repair_reason"),
            },
            "vocabulary": {
                "measures": _json_list(document.get("measures_json")),
                "dimension_names": _json_list(document.get("dimension_names_json")),
                "dimension_values": _json_list(document.get("dimension_values_json")),
                "units": _json_list(document.get("units_json")),
                "other_ambiguous": _json_list(document.get("other_ambiguous_json")),
                "domains": _json_list(document.get("domains_json")),
            },
            "geographies": _json_list(document.get("geographies_json")),
            "times": _json_list(document.get("times_json")),
            "evidence": _json_list(document.get("evidence_json")),
            "related_terms": related_terms,
        }

    def term_detail(self, term_id: str) -> dict[str, Any] | None:
        term = self.terms.get(term_id)
        if term is None:
            return None
        output = self.term_outputs.get(term_id, {})
        relations = self.relations_by_term.get(term_id, [])
        incoming = [relation for relation in relations if relation.get("target_term_id") == term_id]
        outgoing = [relation for relation in relations if relation.get("source_term_id") == term_id]
        return {
            "term_id": term_id,
            "canonical_term": str(term.get("canonical_term") or output.get("canonical_term") or ""),
            "category": output.get("category"),
            "confidence": _float_or_none(output.get("confidence")),
            "provenance": {
                "matching_key": term.get("matching_key"),
                "table_count": term.get("table_count"),
                "occurrence_count": term.get("occurrence_count"),
                "classification_variant": output.get("variant"),
                "protected": _bool_or_none(output.get("protected")),
                "evidence": output.get("evidence"),
                "run_id": output.get("run_id") or term.get("run_id"),
            },
            "occurrence_ids": [str(item) for item in _json_list(term.get("occurrence_ids_json"))],
            "table_appearances": self.term_tables.get(term_id, []),
            "cluster": self.cluster_by_term.get(term_id),
            "relations": {"incoming": incoming, "outgoing": outgoing},
        }

    def term_list(
        self,
        *,
        page: int,
        page_size: int,
        category: str | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        query = (q or "").strip().casefold()
        rows: list[dict[str, Any]] = []
        for term_id, term in self.terms.items():
            output = self.term_outputs.get(term_id, {})
            item_category = output.get("category")
            canonical_term = str(term.get("canonical_term") or output.get("canonical_term") or "")
            if category and item_category != category:
                continue
            if query and query not in canonical_term.casefold():
                continue
            relations = self.relations_by_term.get(term_id, [])
            incoming_count = sum(
                1 for relation in relations if relation.get("target_term_id") == term_id
            )
            outgoing_count = sum(
                1 for relation in relations if relation.get("source_term_id") == term_id
            )
            rows.append(
                {
                    "term_id": term_id,
                    "canonical_term": canonical_term,
                    "category": item_category,
                    "confidence": _float_or_none(output.get("confidence")),
                    "table_count": term.get("table_count"),
                    "occurrence_count": term.get("occurrence_count"),
                    "cluster": self.cluster_by_term.get(term_id),
                    "relation_counts": {
                        "incoming": incoming_count,
                        "outgoing": outgoing_count,
                        "total": incoming_count + outgoing_count,
                    },
                }
            )
        rows.sort(
            key=lambda item: (
                str(item.get("category") or ""),
                str(item.get("canonical_term") or "").casefold(),
                str(item.get("term_id") or ""),
            )
        )
        items, total = _page(rows, page=page, page_size=page_size)
        return {
            "pagination": {"page": page, "page_size": page_size, "total": total},
            "items": items,
        }

    def cluster_list(self, *, page: int, page_size: int) -> dict[str, Any]:
        items, total = _page(self.clusters, page=page, page_size=page_size)
        return {
            "pagination": {"page": page, "page_size": page_size, "total": total},
            "items": items,
        }

    def relation_list(
        self,
        *,
        page: int,
        page_size: int,
        term_id: str | None = None,
        relation_type: str | None = None,
    ) -> dict[str, Any]:
        rows = self.relations
        if term_id:
            rows = [
                row
                for row in rows
                if row.get("source_term_id") == term_id or row.get("target_term_id") == term_id
            ]
        if relation_type:
            rows = [row for row in rows if row.get("relation_type") == relation_type]
        items, total = _page(rows, page=page, page_size=page_size)
        return {
            "pagination": {"page": page, "page_size": page_size, "total": total},
            "items": items,
        }


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _bool_or_none(value: object) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).casefold() == "true"


def _load_terms(config: AppConfig) -> dict[str, dict[str, Any]]:
    rows = _read_required_parquet(config.paths.processed_dir / "vocabulary.parquet")
    return {str(row["term_id"]): row for row in rows}


def _load_term_outputs(config: AppConfig) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for category, filename in CSV_BY_CATEGORY.items():
        for row in _read_required_csv(config.paths.outputs_dir / filename):
            term_id = row.get("term_id", "")
            if term_id:
                row["category"] = row.get("category") or category.value
                outputs[term_id] = row
    return outputs


def _load_table_terms(
    config: AppConfig,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_term: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_required_parquet(config.paths.processed_dir / "table_vocabulary.parquet"):
        table_id = str(row["table_id"])
        term_id = str(row["term_id"])
        item = {
            "term_id": term_id,
            "canonical_term": row.get("canonical_term"),
            "occurrence_count": row.get("occurrence_count"),
            "source_areas": _json_list(row.get("source_areas_json")),
        }
        by_table[table_id].append(item)
        by_term[term_id].append({"table_id": table_id, **item})
    for rows in by_table.values():
        rows.sort(key=lambda item: (str(item["canonical_term"]).casefold(), str(item["term_id"])))
    for rows in by_term.values():
        rows.sort(key=lambda item: str(item["table_id"]))
    return dict(by_table), dict(by_term)


def _load_clusters(config: AppConfig) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = _read_required_csv(config.paths.outputs_dir / "measure_clusters.csv")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("cluster_id", "")].append(row)

    clusters: list[dict[str, Any]] = []
    by_term: dict[str, dict[str, Any]] = {}
    for cluster_id, members in grouped.items():
        domain_counts = Counter(row.get("domain", "") for row in members if row.get("domain"))
        domain = domain_counts.most_common(1)[0][0] if domain_counts else ""
        representatives = [
            {"term_id": row["term_id"], "term": row.get("term", "")}
            for row in members
            if str(row.get("is_representative", "")).casefold() == "true"
        ]
        member_items = [
            {
                "term_id": row.get("term_id", ""),
                "term": row.get("term", ""),
                "membership_probability": _float_or_none(row.get("membership_probability")),
                "is_representative": _bool_or_none(row.get("is_representative")) or False,
            }
            for row in sorted(members, key=lambda item: str(item.get("term", "")).casefold())
        ]
        cluster = {
            "cluster_id": cluster_id,
            "domain": domain,
            "size": len(members),
            "representatives": representatives[:10],
            "members": member_items[:25],
        }
        clusters.append(cluster)
        for row in members:
            term_id = row.get("term_id", "")
            if term_id:
                by_term[term_id] = {
                    "cluster_id": cluster_id,
                    "domain": domain,
                    "membership_probability": _float_or_none(row.get("membership_probability")),
                    "is_representative": _bool_or_none(row.get("is_representative")) or False,
                }

    clusters.sort(key=lambda item: (str(item["domain"]).casefold(), str(item["cluster_id"])))
    return clusters, by_term


def _load_relations(
    config: AppConfig,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = []
    by_term: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_required_csv(config.paths.outputs_dir / "measure_relations.csv"):
        item = {
            "relation_id": row.get("relation_id", ""),
            "source_term_id": row.get("source_term_id", ""),
            "source_term": row.get("source_term", ""),
            "target_term_id": row.get("target_term_id", ""),
            "target_term": row.get("target_term", ""),
            "relation_type": row.get("relation_type", ""),
            "confidence": _float_or_none(row.get("confidence")),
            "evidence_ids": _json_list(row.get("evidence_ids_json")),
            "evidence": _json_dict(row.get("evidence")),
            "generation_methods": _json_list(row.get("generation_methods_json")),
            "run_id": row.get("run_id", ""),
        }
        rows.append(item)
        if item["source_term_id"]:
            by_term[str(item["source_term_id"])].append(item)
        if item["target_term_id"]:
            by_term[str(item["target_term_id"])].append(item)
    rows.sort(key=lambda item: (-float(item.get("confidence") or 0.0), str(item["relation_id"])))
    return rows, dict(by_term)


def _load_evaluation(config: AppConfig) -> dict[str, dict[str, Any]]:
    specs = {
        "extraction": config.paths.reports_dir / "extraction_metrics.json",
        "classification": config.paths.reports_dir / "classification_metrics.json",
        "clustering": config.paths.reports_dir / "clustering_metrics.json",
        "relations": config.paths.reports_dir / "relations_metrics.json",
        "retrieval": config.paths.reports_dir / "retrieval_metrics.json",
    }
    summaries: dict[str, dict[str, Any]] = {}
    for area, path in specs.items():
        if path.exists():
            summaries[area] = {"status": "available", "path": str(path), "data": _read_json(path)}
        else:
            summaries[area] = {"status": "missing", "path": str(path), "data": {}}

    latest_clustering = _latest_json(
        config.paths.outputs_dir / "clustering",
        "*/clustering_summary.json",
    )
    latest_relations = _latest_json(
        config.paths.outputs_dir / "relations",
        "*/relation_summary.json",
    )
    if latest_clustering and summaries["clustering"]["status"] == "missing":
        summaries["clustering"] = {
            "status": "available",
            "path": str(latest_clustering),
            "data": _read_json(latest_clustering),
        }
    if latest_relations and summaries["relations"]["status"] == "missing":
        summaries["relations"] = {
            "status": "available",
            "path": str(latest_relations),
            "data": _read_json(latest_relations),
        }
    return summaries


def load_api_state(settings: ApiSettings) -> ApiState:
    """Load and validate the API artifact set."""

    config = load_config(settings.config_path)
    validate_required_artifact_paths(config)
    search_validation = validate_search_artifacts(config)
    search_engine = SearchEngine(config)
    table_rows = _read_required_parquet(config.paths.processed_dir / "tables.parquet")
    tables = {str(row["table_id"]): row for row in table_rows}
    documents = search_engine.documents
    terms = _load_terms(config)
    term_outputs = _load_term_outputs(config)
    table_terms, term_tables = _load_table_terms(config)
    clusters, cluster_by_term = _load_clusters(config)
    relations, relations_by_term = _load_relations(config)
    validate_loaded_artifacts(
        tables=tables,
        search_documents=documents,
        terms=terms,
        term_outputs=term_outputs,
        table_terms=table_terms,
        clusters=clusters,
        cluster_by_term=cluster_by_term,
        relations=relations,
    )
    evaluation = _load_evaluation(config)
    warnings = [
        f"{area} evaluation summary is missing"
        for area, summary in evaluation.items()
        if summary.get("status") == "missing"
    ]
    return ApiState(
        config=config,
        search_engine=search_engine,
        search_validation=search_validation,
        loaded_at=datetime.now(UTC),
        warnings=warnings,
        tables=tables,
        search_documents=documents,
        terms=terms,
        term_outputs=term_outputs,
        table_terms=table_terms,
        term_tables=term_tables,
        clusters=clusters,
        cluster_by_term=cluster_by_term,
        relations=relations,
        relations_by_term=relations_by_term,
        evaluation=evaluation,
    )
