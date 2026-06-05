"""Evaluation utilities for Milestone 5 measure relationships."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from statvocab.config import AppConfig
from statvocab.contracts import RelationType


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _latest_review_path(config: AppConfig) -> Path | None:
    candidates = sorted(
        (config.paths.outputs_dir / "relations").glob("*/manual_relation_review_sample.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _latest_summary(config: AppConfig) -> dict[str, Any]:
    candidates = sorted(
        (config.paths.outputs_dir / "relations").glob("*/relation_summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return {}
    return cast(dict[str, Any], json.loads(candidates[-1].read_text(encoding="utf-8")))


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "y"}


def _manual_metrics(config: AppConfig) -> dict[str, Any]:
    path = _latest_review_path(config)
    if path is None:
        return {"status": "pending", "message": "Manual relation review sample is not available."}
    rows = _read_csv(path)
    reviewed = [row for row in rows if row.get("is_valid_relation", "").strip()]
    payload: dict[str, Any] = {
        "status": "available" if reviewed else "pending",
        "review_sample": str(path),
        "sample_count": len(rows),
        "completed_count": len(reviewed),
    }
    if not reviewed:
        return payload

    valid = [row for row in reviewed if _truthy(row.get("is_valid_relation", ""))]
    typed_correct = [
        row
        for row in valid
        if _truthy(row.get("correct_relation_type", ""))
        or row.get("gold_relation_type", "") == row.get("relation_type", "")
    ]
    payload["precision_at_10"] = _precision_at(rows, 10)
    payload["precision_at_25"] = _precision_at(rows, 25)
    payload["precision_at_50"] = _precision_at(rows, 50)
    payload["typed_accuracy"] = len(typed_correct) / len(valid) if valid else 0.0
    payload["false_positive_taxonomy"] = dict(
        sorted(
            Counter(
                row.get("false_positive_type", "").strip() or "unspecified"
                for row in reviewed
                if not _truthy(row.get("is_valid_relation", ""))
            ).items()
        )
    )
    return payload


def _precision_at(rows: list[dict[str, str]], k: int) -> float | None:
    top = rows[: min(k, len(rows))]
    reviewed = [row for row in top if row.get("is_valid_relation", "").strip()]
    if not reviewed:
        return None
    valid = sum(1 for row in reviewed if _truthy(row.get("is_valid_relation", "")))
    return valid / len(reviewed)


def evaluate_relations(
    config: AppConfig,
    *,
    output_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Evaluate relation artifacts and manual review status."""

    relation_path = config.paths.outputs_dir / "measure_relations.csv"
    relation_rows = _read_csv(relation_path)
    measure_rows = _read_csv(config.paths.outputs_dir / "measures.csv")
    valid_measure_ids = {row.get("term_id", "") for row in measure_rows}
    allowed_types = {relation_type.value for relation_type in RelationType}

    failures: list[str] = []
    relation_ids = [row.get("relation_id", "") for row in relation_rows]
    duplicate_ids = sorted(
        relation_id for relation_id, count in Counter(relation_ids).items() if count > 1
    )
    if duplicate_ids:
        failures.append("duplicate relation IDs: " + ", ".join(duplicate_ids[:10]))

    pair_keys = [
        tuple(sorted((row.get("source_term_id", ""), row.get("target_term_id", ""))))
        for row in relation_rows
    ]
    duplicate_pairs = sorted(pair for pair, count in Counter(pair_keys).items() if count > 1)
    if duplicate_pairs:
        failures.append(
            "duplicate relation pairs: "
            + ", ".join("/".join(pair) for pair in duplicate_pairs[:10])
        )

    unknown_ids = sorted(
        {
            term_id
            for row in relation_rows
            for term_id in (row.get("source_term_id", ""), row.get("target_term_id", ""))
            if valid_measure_ids and term_id not in valid_measure_ids
        }
    )
    if unknown_ids:
        failures.append("relations reference unknown measures: " + ", ".join(unknown_ids[:10]))

    self_relations = [
        row.get("relation_id", "")
        for row in relation_rows
        if row.get("source_term_id") == row.get("target_term_id")
    ]
    if self_relations:
        failures.append("self-relations present: " + ", ".join(self_relations[:10]))

    invalid_types = sorted({row.get("relation_type", "") for row in relation_rows} - allowed_types)
    if invalid_types:
        failures.append("invalid relation types: " + ", ".join(invalid_types))

    missing_grounding = [
        row.get("relation_id", "")
        for row in relation_rows
        if not row.get("evidence_ids_json") or not row.get("evidence") or not row.get("confidence")
    ]
    if missing_grounding:
        failures.append(
            "relations missing evidence or confidence: " + ", ".join(missing_grounding[:10])
        )

    type_counts = Counter(row.get("relation_type", "") for row in relation_rows)
    payload: dict[str, Any] = {
        "area": "relations",
        "config_name": config.config_name,
        "corpus": config.corpus.name,
        "artifact": str(relation_path),
        "artifact_status": "available" if relation_rows else "pending",
        "relation_count": len(relation_rows),
        "relation_type_distribution": dict(sorted(type_counts.items())),
        "manual_review": _manual_metrics(config),
        "latest_run_summary": _latest_summary(config),
        "validation": {"passed": not failures, "failures": failures},
    }
    metrics_path = output_path or config.paths.reports_dir / "relations_metrics.json"
    return _write_json(payload, metrics_path), payload
