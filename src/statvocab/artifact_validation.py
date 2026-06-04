"""Artifact validation for Milestone 3 and later outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from statvocab.classification import CSV_BY_CATEGORY
from statvocab.config import AppConfig
from statvocab.contracts import VocabularyCategory


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
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


def validate_artifacts(config: AppConfig, *, run_id: str) -> dict[str, Any]:
    """Validate required semantic partition outputs."""

    manifest = _read_manifest(config, run_id)
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
