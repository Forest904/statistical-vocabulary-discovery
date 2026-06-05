"""Feature generation and annotation sampling for Milestone 3 classification."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from statvocab.config import AppConfig
from statvocab.contracts import VocabularyCategory

CATEGORY_VALUES = tuple(category.value for category in VocabularyCategory)
GOLD_FIELDNAMES = [
    "term_id",
    "canonical_term",
    "matching_key",
    "split",
    "stratum",
    "table_count",
    "occurrence_count",
    "role_summary_json",
    "feature_summary_json",
    "category",
    "annotator_id",
    "notes",
]


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read a Parquet file into dictionaries, failing clearly when missing."""

    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the previous pipeline stage first.")
    return cast(list[dict[str, Any]], pq.read_table(path).to_pylist())


def write_parquet_rows(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write rows to Parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def write_csv_rows(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> Path:
    """Write rows to a stable CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows; return an empty list when the file is absent."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _json_dict(value: object) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _freq_bin(count: int) -> str:
    if count <= 1:
        return "occ_1"
    if count <= 5:
        return "occ_2_5"
    if count <= 20:
        return "occ_6_20"
    return "occ_21_plus"


def _table_bin(count: int) -> str:
    if count <= 1:
        return "tables_1"
    if count <= 5:
        return "tables_2_5"
    if count <= 20:
        return "tables_6_20"
    return "tables_21_plus"


def _shape_flags(term: str) -> dict[str, bool]:
    lower = term.casefold()
    return {
        "has_percent": "%" in term or "percentage" in lower or "percent" in lower,
        "has_digit": any(char.isdigit() for char in term),
        "has_age_pattern": bool(re.search(r"\b\d+\s*(?:to|-)\s*\d+\s*years?\b", lower))
        or "years or over" in lower,
        "is_short_code": bool(re.fullmatch(r"[A-Z0-9_]{1,6}", term.strip())),
        "has_unit_word": any(
            word in lower
            for word in (
                "euro",
                "tonne",
                "kilogram",
                "hectare",
                "person",
                "inhabitant",
                "hour",
                "litre",
                "index",
                "number",
                "rate",
                "percentage",
            )
        ),
        "has_value_word": any(
            word in lower
            for word in (
                "total",
                "male",
                "female",
                "annual",
                "monthly",
                "quarterly",
                "employed",
                "unemployed",
                "from ",
                "less than",
                "over",
            )
        ),
        "looks_like_dimension_name": any(
            word in lower
            for word in ("unit of measure", "time frequency", "classification", "category")
        )
        or "\\" in term,
    }


def feature_rows(config: AppConfig) -> list[dict[str, Any]]:
    """Build deterministic structural and lexical features for every vocabulary term."""

    vocabulary = read_parquet_rows(config.paths.processed_dir / "vocabulary.parquet")
    rows: list[dict[str, Any]] = []
    for row in sorted(vocabulary, key=lambda item: str(item["matching_key"])):
        role_summary = _json_dict(row.get("role_summary_json"))
        feature_summary = _json_dict(row.get("feature_summary_json"))
        metadata_columns = _json_list(feature_summary.get("metadata_columns"))
        canonical = str(row["canonical_term"])
        table_count = int(row.get("table_count") or 0)
        occurrence_count = int(row.get("occurrence_count") or 0)
        role_names = sorted(str(key) for key in role_summary)
        has_title = bool(feature_summary.get("has_title_evidence"))
        has_header = bool(feature_summary.get("has_header_evidence"))
        has_metadata_value = bool(feature_summary.get("has_metadata_value_evidence"))
        flags = _shape_flags(canonical)
        stratum = "|".join(
            [
                "+".join(role_names) or "none",
                _freq_bin(occurrence_count),
                _table_bin(table_count),
                "title" if has_title else "no_title",
                "header" if has_header else "no_header",
                "value" if has_metadata_value else "no_value",
                "conflict" if len(role_names) > 1 else "single_role",
            ]
        )
        rows.append(
            {
                "term_id": row["term_id"],
                "canonical_term": canonical,
                "matching_key": row["matching_key"],
                "table_count": table_count,
                "occurrence_count": occurrence_count,
                "occurrence_ids_json": str(row.get("occurrence_ids_json") or "[]"),
                "role_summary_json": json.dumps(role_summary, sort_keys=True),
                "feature_summary_json": json.dumps(feature_summary, sort_keys=True),
                "metadata_columns_json": json.dumps(metadata_columns, sort_keys=True),
                "source_roles": "+".join(role_names),
                "has_title_evidence": has_title,
                "has_header_evidence": has_header,
                "has_metadata_value_evidence": has_metadata_value,
                "has_conflicting_roles": len(role_names) > 1,
                "frequency_bin": _freq_bin(occurrence_count),
                "table_count_bin": _table_bin(table_count),
                "stratum": stratum,
                **flags,
                "run_id": row["run_id"],
            }
        )
    return rows


def write_classification_features(config: AppConfig) -> Path:
    """Write the feature table used by rules and local models."""

    return write_parquet_rows(
        feature_rows(config),
        config.paths.processed_dir / "classification_features.parquet",
    )


def _split_for_index(index: int, total: int, config: AppConfig) -> str:
    train_cut = round(total * config.classification.train_dev_fraction)
    validation_cut = train_cut + round(total * config.classification.validation_fraction)
    if index < train_cut:
        return "train_dev"
    if index < validation_cut:
        return "validation"
    return "final_test"


def _sample_terms(rows: list[dict[str, Any]], config: AppConfig) -> list[dict[str, Any]]:
    sample_size = min(config.classification.gold_sample_size, len(rows))
    rng = random.Random(config.random_seed)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row["stratum"])].append(row)
    for stratum_rows in by_stratum.values():
        stratum_rows.sort(key=lambda row: str(row["term_id"]))

    selected: dict[str, dict[str, Any]] = {}
    strata = sorted(by_stratum)
    while len(selected) < sample_size and strata:
        progressed = False
        for stratum in list(strata):
            candidates = [row for row in by_stratum[stratum] if str(row["term_id"]) not in selected]
            if not candidates:
                strata.remove(stratum)
                continue
            row = rng.choice(candidates)
            selected[str(row["term_id"])] = row
            progressed = True
            if len(selected) >= sample_size:
                break
        if not progressed:
            break

    return sorted(selected.values(), key=lambda row: str(row["term_id"]))


def ensure_gold_templates(config: AppConfig) -> tuple[Path, Path, Path]:
    """Create deterministic gold sample, label, and blind relabel templates."""

    rows = feature_rows(config)
    sample_rows = _sample_terms(rows, config)
    total = len(sample_rows)
    label_rows: list[dict[str, Any]] = []
    for index, row in enumerate(sample_rows):
        label_rows.append(
            {
                **row,
                "split": _split_for_index(index, total, config),
                "category": "",
                "annotator_id": "",
                "notes": "",
            }
        )

    sample_path = write_csv_rows(
        label_rows,
        config.evaluation.extraction_gold_dir / "vocabulary_gold_sample.csv",
        GOLD_FIELDNAMES,
    )
    labels_path = config.evaluation.extraction_gold_dir / "vocabulary_gold_labels.csv"
    if not labels_path.exists():
        write_csv_rows(label_rows, labels_path, GOLD_FIELDNAMES)

    relabel_count = round(total * config.classification.blind_relabel_fraction)
    rng = random.Random(config.random_seed + 1)
    relabel_rows = sorted(
        rng.sample(label_rows, k=min(relabel_count, total)),
        key=lambda row: str(row["term_id"]),
    )
    relabel_path = config.evaluation.extraction_gold_dir / "vocabulary_gold_relabel.csv"
    if not relabel_path.exists():
        write_csv_rows(relabel_rows, relabel_path, GOLD_FIELDNAMES)
    return sample_path, labels_path, relabel_path


def completed_gold_labels(config: AppConfig) -> list[dict[str, str]]:
    """Return rows whose category is filled with an allowed label."""

    labels_path = config.evaluation.extraction_gold_dir / "vocabulary_gold_labels.csv"
    rows = read_csv_rows(labels_path)
    allowed = set(CATEGORY_VALUES)
    return [row for row in rows if row.get("category", "").strip() in allowed]
