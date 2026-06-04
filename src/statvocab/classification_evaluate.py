"""Evaluation utilities for Milestone 3 semantic classification."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from statvocab.classification_features import (
    CATEGORY_VALUES,
    completed_gold_labels,
    ensure_gold_templates,
    read_csv_rows,
)
from statvocab.config import AppConfig


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _score_rows(
    gold_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_term = {str(row["term_id"]): str(row["category"]) for row in prediction_rows}
    labels = list(CATEGORY_VALUES)
    confusion = {true: dict.fromkeys(labels, 0) for true in labels}
    missing_predictions: list[str] = []
    for row in gold_rows:
        term_id = row["term_id"]
        true_label = row["category"]
        predicted = by_term.get(term_id)
        if predicted is None:
            missing_predictions.append(term_id)
            continue
        if true_label in confusion and predicted in confusion[true_label]:
            confusion[true_label][predicted] += 1

    total = sum(sum(preds.values()) for preds in confusion.values())
    correct = sum(confusion[label][label] for label in labels)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    weighted_f1 = (
        sum(float(per_class[label]["f1"]) * int(per_class[label]["support"]) for label in labels)
        / total
        if total
        else 0.0
    )
    return {
        "gold_count": len(gold_rows),
        "scored_count": total,
        "missing_predictions": missing_predictions,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _cohen_kappa(primary: list[str], secondary: list[str]) -> float:
    total = len(primary)
    if total == 0:
        return 0.0
    agreement = (
        sum(1 for left, right in zip(primary, secondary, strict=True) if left == right) / total
    )
    primary_counts = Counter(primary)
    secondary_counts = Counter(secondary)
    chance = sum(
        (primary_counts[label] / total) * (secondary_counts[label] / total)
        for label in set(primary_counts) | set(secondary_counts)
    )
    if chance == 1.0:
        return 1.0
    return (agreement - chance) / (1.0 - chance)


def _agreement(config: AppConfig) -> dict[str, Any]:
    label_path = config.evaluation.extraction_gold_dir / "vocabulary_gold_labels.csv"
    relabel_path = config.evaluation.extraction_gold_dir / "vocabulary_gold_relabel.csv"
    labels = {
        row["term_id"]: row
        for row in read_csv_rows(label_path)
        if row.get("category") in CATEGORY_VALUES
    }
    relabels = [
        row
        for row in read_csv_rows(relabel_path)
        if row.get("category") in CATEGORY_VALUES and row.get("term_id") in labels
    ]
    if not relabels:
        return {"status": "pending", "message": "Blind relabel categories are not complete."}

    primary = [labels[row["term_id"]]["category"] for row in relabels]
    secondary = [row["category"] for row in relabels]
    disagreements = [
        {
            "term_id": row["term_id"],
            "canonical_term": row["canonical_term"],
            "primary_category": labels[row["term_id"]]["category"],
            "relabel_category": row["category"],
            "notes": row.get("notes", ""),
        }
        for row in relabels
        if labels[row["term_id"]]["category"] != row["category"]
    ]
    return {
        "status": "available",
        "reviewed_count": len(relabels),
        "raw_agreement": sum(
            1 for left, right in zip(primary, secondary, strict=True) if left == right
        )
        / len(relabels),
        "cohen_kappa": _cohen_kappa(primary, secondary),
        "disagreements": disagreements,
    }


def evaluate_classification(
    config: AppConfig,
    *,
    prediction_rows: list[dict[str, Any]] | None = None,
    variant: str | None = None,
    output_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Evaluate available classification predictions and gold labels."""

    sample_path, labels_path, relabel_path = ensure_gold_templates(config)
    gold_rows = completed_gold_labels(config)
    payload: dict[str, Any] = {
        "area": "classification",
        "config_name": config.config_name,
        "corpus": config.corpus.name,
        "gold_sample": str(sample_path),
        "gold_labels": str(labels_path),
        "gold_relabel": str(relabel_path),
        "gold_status": "available" if gold_rows else "pending",
        "agreement": _agreement(config),
    }
    if variant is not None:
        payload["variant"] = variant
    if prediction_rows is not None and gold_rows:
        payload["metrics"] = _score_rows(gold_rows, prediction_rows)
        final_test_gold = [row for row in gold_rows if row.get("split") == "final_test"]
        validation_gold = [row for row in gold_rows if row.get("split") == "validation"]
        payload["validation_metrics"] = _score_rows(validation_gold, prediction_rows)
        payload["final_test_metrics"] = _score_rows(final_test_gold, prediction_rows)
    elif prediction_rows is not None:
        payload["metrics_status"] = "pending_gold_labels"
        payload["message"] = "Classification predictions exist, but gold labels are pending."
    else:
        payload["metrics_status"] = "pending_predictions"

    metrics_path = output_path or config.paths.reports_dir / "classification_metrics.json"
    return _write_json(payload, metrics_path), payload
