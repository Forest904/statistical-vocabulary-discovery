"""Targeted review sampling for conservative vocabulary reclaim audits."""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from statvocab.classification_features import GOLD_FIELDNAMES, read_csv_rows, write_csv_rows
from statvocab.config import AppConfig

TARGETED_SAMPLE = "vocabulary_reclaim_sample.csv"
TARGETED_LABELS = "vocabulary_reclaim_labels.csv"
TARGETED_RELABEL = "vocabulary_reclaim_relabel.csv"
TARGETED_FIELDNAMES = [*GOLD_FIELDNAMES, "audit_source", "target_cohort"]


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    _raise_csv_field_limit()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _feature_rows(config: AppConfig) -> dict[str, dict[str, Any]]:
    path = config.paths.processed_dir / "classification_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run extraction/classification features first.")
    return {str(row["term_id"]): row for row in pq.read_table(path).to_pylist()}


def _json_dict(value: object) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, dict) else {}


def _noise_like(term: str) -> bool:
    stripped = term.strip()
    if not stripped:
        return True
    if stripped in {"-", ",", ".", ":", ";"}:
        return True
    if re.fullmatch(r"[-?,.\\/_]+", stripped):
        return True
    replacement_mark_count = stripped.count("?")
    return replacement_mark_count >= 2 or replacement_mark_count >= max(1, len(stripped) // 4)


def _cohort(row: dict[str, str], feature: dict[str, Any]) -> str:
    roles = str(feature.get("source_roles") or "")
    term = row["canonical_term"]
    if roles == "title_full":
        return "title_full"
    if roles == "title_clause":
        return "title_clause"
    if bool(feature.get("has_conflicting_roles")) or _noise_like(term) or bool(
        feature.get("is_short_code")
    ):
        return "short_code_conflict_noise"
    if roles == "metadata_value" and (
        bool(feature.get("has_unit_word")) or bool(feature.get("has_digit"))
    ):
        return "unit_range_digit_metadata_value"
    if roles == "metadata_value":
        return "frequent_value_metadata_value"
    return "short_code_conflict_noise"


TARGET_COUNTS = {
    "title_full": 80,
    "title_clause": 40,
    "frequent_value_metadata_value": 60,
    "unit_range_digit_metadata_value": 40,
    "short_code_conflict_noise": 30,
}


def _split_for_index(index: int, config: AppConfig) -> str:
    train_cut = config.classification.targeted_train_dev_count
    validation_cut = train_cut + config.classification.targeted_validation_count
    if index < train_cut:
        return "train_dev"
    if index < validation_cut:
        return "validation"
    return "final_test"


def _sample_other_rows(config: AppConfig) -> list[dict[str, Any]]:
    other_path = config.paths.outputs_dir / "other_ambiguous.csv"
    features = _feature_rows(config)
    original_sample_path = config.evaluation.extraction_gold_dir / "vocabulary_gold_sample.csv"
    original_sample_ids = {row["term_id"] for row in read_csv_rows(original_sample_path)}
    rows_by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(other_path):
        term_id = row["term_id"]
        if term_id in original_sample_ids:
            continue
        feature = features.get(term_id)
        if feature is None:
            continue
        cohort = _cohort(row, feature)
        rows_by_cohort[cohort].append({**feature, "target_cohort": cohort})

    rng = random.Random(config.random_seed + 7)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for cohort, target_count in TARGET_COUNTS.items():
        candidates = sorted(
            rows_by_cohort.get(cohort, []),
            key=lambda row: (
                -int(row.get("occurrence_count") or 0),
                str(row["canonical_term"]).casefold(),
                str(row["term_id"]),
            ),
        )
        if len(candidates) > target_count:
            head = candidates[: target_count // 2]
            tail = candidates[target_count // 2 :]
            sampled_tail = rng.sample(tail, k=min(target_count - len(head), len(tail)))
            candidates = [*head, *sampled_tail]
        for row in candidates[:target_count]:
            selected.append(row)
            selected_ids.add(str(row["term_id"]))

    if len(selected) < config.classification.targeted_reclaim_sample_size:
        fallback = [
            row
            for rows in rows_by_cohort.values()
            for row in rows
            if str(row["term_id"]) not in selected_ids
        ]
        fallback.sort(key=lambda row: str(row["term_id"]))
        needed = config.classification.targeted_reclaim_sample_size - len(selected)
        selected.extend(rng.sample(fallback, k=min(needed, len(fallback))))

    selected.sort(key=lambda row: (str(row["target_cohort"]), str(row["term_id"])))
    selected = selected[: config.classification.targeted_reclaim_sample_size]
    for index, row in enumerate(selected):
        row["split"] = _split_for_index(index, config)
        row["category"] = ""
        row["annotator_id"] = ""
        row["notes"] = ""
        row["audit_source"] = "targeted_reclaim"
    return selected


def ensure_targeted_reclaim_templates(config: AppConfig) -> tuple[Path, Path, Path]:
    """Create deterministic targeted reclaim templates without overwriting labels."""

    rows = _sample_other_rows(config)
    gold_dir = config.evaluation.extraction_gold_dir
    sample_path = gold_dir / TARGETED_SAMPLE
    labels_path = gold_dir / TARGETED_LABELS
    relabel_path = gold_dir / TARGETED_RELABEL
    write_csv_rows(rows, sample_path, TARGETED_FIELDNAMES)
    if not labels_path.exists():
        write_csv_rows(rows, labels_path, TARGETED_FIELDNAMES)

    relabel_count = min(config.classification.targeted_relabel_size, len(rows))
    rng = random.Random(config.random_seed + 8)
    relabel_rows = sorted(
        rng.sample(rows, k=relabel_count),
        key=lambda row: str(row["term_id"]),
    )
    if not relabel_path.exists():
        write_csv_rows(relabel_rows, relabel_path, TARGETED_FIELDNAMES)
    return sample_path, labels_path, relabel_path


def completed_targeted_labels(config: AppConfig) -> list[dict[str, str]]:
    """Return completed targeted labels with source metadata."""

    rows = read_csv_rows(config.evaluation.extraction_gold_dir / TARGETED_LABELS)
    completed = []
    for row in rows:
        if row.get("category", "").strip():
            row["audit_source"] = row.get("audit_source") or "targeted_reclaim"
            completed.append(row)
    return completed
