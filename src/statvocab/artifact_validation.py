"""Artifact validation for Milestone 3 and later outputs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from statvocab.classification import CSV_BY_CATEGORY
from statvocab.config import AppConfig
from statvocab.contracts import VocabularyCategory
from statvocab.resources import file_md5

CORE_RELEASE_FILES = (
    "outputs/measures.csv",
    "outputs/dimension_names.csv",
    "outputs/dimension_values.csv",
    "outputs/units.csv",
    "outputs/other_ambiguous.csv",
    "outputs/measure_clusters.csv",
    "outputs/measure_relations.csv",
    "outputs/search/current_index.json",
    "data/processed/tables.parquet",
    "data/processed/table_times.parquet",
    "data/processed/table_strings.parquet",
    "data/processed/table_geographies.parquet",
    "data/processed/title_terms.parquet",
    "data/processed/term_occurrences.parquet",
    "data/processed/table_vocabulary.parquet",
    "data/processed/vocabulary.parquet",
    "report/extraction_metrics.json",
    "report/classification_metrics.json",
    "report/clustering_metrics.json",
    "report/relations_metrics.json",
    "report/retrieval_metrics.json",
    "report/final_report.md",
    "report/final_report.pdf",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _read_occurrence_ids(config: AppConfig) -> set[str]:
    path = config.paths.processed_dir / "term_occurrences.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing provenance artifact: {path}")
    rows = pq.read_table(path, columns=["occurrence_id"]).to_pylist()
    return {str(row["occurrence_id"]) for row in rows}


def _read_manifest(config: AppConfig, run_id: str) -> dict[str, Any]:
    manifest_dir = config.paths.outputs_dir / "manifests"
    candidates = sorted(manifest_dir.glob(f"{run_id}_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No run manifest found for run ID {run_id}")
    return cast(dict[str, Any], json.loads(candidates[-1].read_text(encoding="utf-8")))


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "md5": file_md5(path) if path.exists() and path.is_file() else None,
    }


def _current_category_export_run_ids(config: AppConfig) -> list[str]:
    """Return run IDs recorded in the current top-level category CSV exports."""

    run_ids: set[str] = set()
    for filename in CSV_BY_CATEGORY.values():
        path = config.paths.outputs_dir / filename
        if not path.exists():
            continue
        for row in _read_csv(path):
            run_id = row.get("run_id", "").strip()
            if run_id:
                run_ids.add(run_id)
    return sorted(run_ids)


def write_core_release_manifest(
    config: AppConfig,
    *,
    output_path: Path | None = None,
) -> Path:
    """Write a checksum manifest for the current core release artifacts."""

    if config.corpus.name != "core":
        raise ValueError("Core release manifests must be generated with the core config.")
    path = output_path or config.paths.reports_dir / "core_release_manifest.json"
    category_export_run_ids = _current_category_export_run_ids(config)
    payload = {
        "category_export_run_ids": category_export_run_ids,
        "config_fingerprint": config.config_fingerprint,
        "config_summary": config.config_summary,
        "run_id": category_export_run_ids[0] if len(category_export_run_ids) == 1 else "mixed",
        "files": [_file_record(Path(relative)) for relative in CORE_RELEASE_FILES],
    }
    files = cast(list[dict[str, Any]], payload["files"])
    missing = [str(item["path"]) for item in files if not item["exists"]]
    if missing:
        raise FileNotFoundError("Missing core release artifacts: " + ", ".join(missing))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_core_release_manifest(
    config: AppConfig,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Validate that frozen core artifacts still match their recorded checksums."""

    path = manifest_path or config.paths.reports_dir / "core_release_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing core release manifest: {path}")
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    failures: list[str] = []
    for item in cast(list[dict[str, Any]], payload.get("files") or []):
        artifact_path = Path(str(item.get("path") or ""))
        if not artifact_path.exists():
            failures.append(f"missing artifact {artifact_path}")
            continue
        expected_size = int(item.get("size_bytes") or 0)
        if artifact_path.stat().st_size != expected_size:
            failures.append(f"size changed for {artifact_path}")
        expected_md5 = item.get("md5")
        if expected_md5 and file_md5(artifact_path) != expected_md5:
            failures.append(f"checksum changed for {artifact_path}")
    if failures:
        raise ValueError("; ".join(failures[:20]))
    return {
        "validated": True,
        "manifest": str(path),
        "file_count": len(payload.get("files") or []),
        "run_id": payload.get("run_id", ""),
    }


def _validate_full_run(
    config: AppConfig,
    *,
    run_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = config.paths.outputs_dir / "full_corpus" / run_id / "stage_checkpoints.json"
    if not checkpoint_path.exists():
        for artifact in manifest.get("artifacts", []):
            candidate = Path(str(artifact))
            if candidate.name == "stage_checkpoints.json":
                checkpoint_path = candidate
                break
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing full-corpus checkpoint artifact: {checkpoint_path}")

    checkpoints = cast(
        list[dict[str, Any]],
        json.loads(checkpoint_path.read_text(encoding="utf-8")),
    )
    failures: list[str] = []
    status_counts: dict[str, int] = {}
    incomplete: list[dict[str, str]] = []
    for checkpoint in checkpoints:
        stage = str(checkpoint.get("stage") or "")
        status = str(checkpoint.get("status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "succeeded":
            for artifact in checkpoint.get("artifacts", []):
                artifact_path = Path(str(artifact.get("path") or ""))
                if not artifact_path.exists():
                    failures.append(f"{stage} declared missing artifact {artifact_path}")
                    continue
                expected_size = int(artifact.get("size_bytes") or 0)
                if artifact_path.is_file() and artifact_path.stat().st_size <= 0:
                    failures.append(f"{stage} artifact is empty: {artifact_path}")
                if expected_size and artifact_path.exists():
                    actual_size = artifact_path.stat().st_size
                    if artifact_path.is_dir():
                        actual_size = sum(
                            child.stat().st_size
                            for child in artifact_path.rglob("*")
                            if child.is_file()
                        )
                    if actual_size != expected_size:
                        failures.append(
                            f"{stage} artifact size changed for {artifact_path}: "
                            f"expected {expected_size}, found {actual_size}"
                        )
                expected_md5 = artifact.get("md5")
                if (
                    expected_md5
                    and artifact_path.is_file()
                    and file_md5(artifact_path) != expected_md5
                ):
                    failures.append(f"{stage} artifact checksum changed: {artifact_path}")
            continue

        message = str(checkpoint.get("failure_message") or "")
        last_valid = checkpoint.get("last_valid_checkpoint")
        if not message:
            failures.append(f"{stage} is incomplete without a blocker message")
        if stage != "disk-preflight" and not last_valid:
            failures.append(f"{stage} is incomplete without a last valid checkpoint")
        incomplete.append(
            {
                "stage": stage,
                "status": status,
                "blocker": message,
                "last_valid_checkpoint": str(last_valid or ""),
            }
        )

    if failures:
        raise ValueError("; ".join(failures[:20]))
    return {
        "run_id": run_id,
        "validated": True,
        "pipeline_stage": "run-all",
        "checkpoint_count": len(checkpoints),
        "stage_status_counts": status_counts,
        "incomplete_stages": incomplete,
    }


def validate_artifacts(config: AppConfig, *, run_id: str) -> dict[str, Any]:
    """Validate required semantic partition outputs."""

    manifest = _read_manifest(config, run_id)
    if str(manifest.get("pipeline_stage", "")) == "run-all":
        return _validate_full_run(config, run_id=run_id, manifest=manifest)
    if not str(manifest.get("pipeline_stage", "")).startswith("classify-"):
        raise ValueError(f"Run {run_id} is not a classification run.")

    occurrence_ids = _read_occurrence_ids(config)
    seen: dict[str, str] = {}
    category_counts: dict[str, int] = {}
    failures: list[str] = []
    for category, filename in CSV_BY_CATEGORY.items():
        path = config.paths.outputs_dir / filename
        rows = _read_csv(path)
        category_counts[category.value] = len(rows)
        terms_for_sort = [(row["canonical_term"].casefold(), row["term_id"]) for row in rows]
        if terms_for_sort != sorted(terms_for_sort):
            failures.append(f"{filename} is not sorted by canonical term and term ID")
        for row in rows:
            term_id = row["term_id"]
            if row.get("category") != category.value:
                failures.append(f"{filename} contains row with category {row.get('category')}")
            if term_id in seen:
                failures.append(f"{term_id} appears in both {seen[term_id]} and {filename}")
            seen[term_id] = filename
            evidence = json.loads(row.get("occurrence_ids_json") or "[]")
            if not evidence:
                failures.append(f"{term_id} has no occurrence provenance")
            unknown = sorted({str(item) for item in evidence} - occurrence_ids)
            if unknown:
                joined = ", ".join(unknown)
                failures.append(f"{term_id} references unknown occurrence IDs: {joined}")

    vocabulary_path = config.paths.processed_dir / "vocabulary.parquet"
    vocabulary_rows = pq.read_table(vocabulary_path, columns=["term_id"]).to_pylist()
    vocabulary_ids = {str(row["term_id"]) for row in vocabulary_rows}
    missing = sorted(vocabulary_ids - set(seen))
    extra = sorted(set(seen) - vocabulary_ids)
    if missing:
        failures.append(f"{len(missing)} vocabulary terms have no final category")
    if extra:
        failures.append(f"{len(extra)} output terms are not in vocabulary")

    allowed_categories = {category.value for category in VocabularyCategory}
    invalid_categories = sorted(set(category_counts) - allowed_categories)
    if invalid_categories:
        failures.append(f"Invalid categories found: {', '.join(invalid_categories)}")
    if failures:
        raise ValueError("; ".join(failures[:20]))

    return {
        "run_id": run_id,
        "validated": True,
        "term_count": len(seen),
        "category_counts": category_counts,
        "accepted_hallucination_count": 0,
    }
