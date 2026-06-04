"""Evaluation support for Milestone 2 extraction artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from statvocab.config import AppConfig
from statvocab.normalize import normalize_matching_key

AREAS = ("time", "string", "geography", "title")


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return cast(list[dict[str, Any]], pq.read_table(path).to_pylist())


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _artifact_sets(processed_dir: Path, geography_variant: str) -> dict[str, set[tuple[str, str]]]:
    time_rows = _read_parquet_rows(processed_dir / "table_times.parquet")
    string_rows = _read_parquet_rows(processed_dir / "table_strings.parquet")
    geography_rows = [
        row
        for row in _read_parquet_rows(processed_dir / "table_geographies.parquet")
        if row.get("variant") == geography_variant
    ]
    title_rows = _read_parquet_rows(processed_dir / "title_terms.parquet")
    return {
        "time": {
            (str(row["table_id"]), normalize_matching_key(str(row["normalized_value"])))
            for row in time_rows
        },
        "string": {
            (str(row["table_id"]), str(row["matching_key"]))
            for row in string_rows
        },
        "geography": {
            (str(row["table_id"]), str(row["matching_key"]))
            for row in geography_rows
        },
        "title": {
            (str(row["table_id"]), str(row["matching_key"]))
            for row in title_rows
        },
    }


def _review_sample(config: AppConfig) -> list[dict[str, Any]]:
    tables = _read_parquet_rows(config.paths.processed_dir / "tables.parquet")
    if not tables:
        return []
    sorted_rows = sorted(
        tables,
        key=lambda row: (
            str(row.get("parse_status") or ""),
            -int(row.get("warning_count") or 0),
            str(row.get("table_id") or ""),
        ),
    )
    sample_size = min(config.extraction.review_sample_size, len(sorted_rows))
    return [
        {
            "table_id": row.get("table_id"),
            "title": row.get("title"),
            "parse_status": row.get("parse_status"),
            "warning_count": row.get("warning_count"),
            "row_count": row.get("row_count"),
            "time_column_count": len(row.get("time_columns") or []),
            "metadata_column_count": len(row.get("metadata_columns") or []),
        }
        for row in sorted_rows[:sample_size]
    ]


def _gold_path(config: AppConfig) -> Path:
    if config.corpus.name == "fixture" and config.corpus.fixture_path is not None:
        return config.corpus.fixture_path / "extraction_gold.csv"
    return config.evaluation.extraction_gold_dir / "extraction_gold_labels.csv"


def _completed_gold_template_path(config: AppConfig, generated_template_path: Path) -> Path:
    if config.corpus.name == "fixture":
        return generated_template_path
    tracked_template_path = config.evaluation.extraction_gold_dir / "extraction_gold_template.csv"
    return tracked_template_path if tracked_template_path.exists() else generated_template_path


def _read_gold(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _score_gold(
    gold_rows: list[dict[str, str]],
    extracted: dict[str, set[tuple[str, str]]],
) -> dict[str, Any]:
    by_area: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    failure_examples: list[dict[str, str]] = []
    for row in gold_rows:
        area = row["area"].strip()
        if area not in AREAS:
            continue
        table_id = row["table_id"].strip()
        key = normalize_matching_key(row["expected_value"])
        expected_present = row.get("expected_present", "true").strip().casefold() != "false"
        actual_present = (table_id, key) in extracted[area]
        if expected_present and actual_present:
            by_area[area]["tp"] += 1
        elif expected_present and not actual_present:
            by_area[area]["fn"] += 1
            failure_examples.append({**row, "error_type": "false_negative"})
        elif not expected_present and actual_present:
            by_area[area]["fp"] += 1
            failure_examples.append({**row, "error_type": "false_positive"})
        else:
            by_area[area]["tn"] += 1

    metrics: dict[str, Any] = {}
    for area in AREAS:
        counts = by_area[area]
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        metrics[area] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    metrics["failure_examples"] = failure_examples[:25]
    return metrics


def run_extraction_evaluation(config: AppConfig) -> tuple[Path, Path, dict[str, Any]]:
    """Create review templates and evaluate extraction when gold labels exist."""

    sample_rows = _review_sample(config)
    review_path = _write_csv(
        sample_rows,
        config.paths.processed_dir / "extraction_review_sample.csv",
        [
            "table_id",
            "title",
            "parse_status",
            "warning_count",
            "row_count",
            "time_column_count",
            "metadata_column_count",
        ],
    )
    template_rows = [
        {
            "table_id": row["table_id"],
            "area": "",
            "expected_value": "",
            "expected_present": "true",
            "notes": "",
        }
        for row in sample_rows
    ]
    template_path = _write_csv(
        template_rows,
        config.paths.processed_dir / "extraction_gold_template.csv",
        ["table_id", "area", "expected_value", "expected_present", "notes"],
    )

    gold_path = _gold_path(config)
    completed_template_path = _completed_gold_template_path(config, template_path)
    gold_rows = _read_gold(gold_path)
    payload: dict[str, Any] = {
        "config_name": config.config_name,
        "corpus": config.corpus.name,
        "review_sample": str(review_path),
        "gold_template": str(completed_template_path),
        "gold_labels": str(gold_path),
        "gold_status": "available" if gold_rows else "pending",
    }
    if gold_rows:
        payload["metrics"] = _score_gold(
            gold_rows,
            _artifact_sets(config.paths.processed_dir, config.extraction.geography_variant),
        )
    else:
        payload["message"] = "Gold labels are pending; annotation template has been written."

    metrics_path = _write_json(payload, config.paths.reports_dir / "extraction_metrics.json")
    return review_path, metrics_path, payload
