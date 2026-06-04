"""Milestone 3 semantic partition runner."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from statvocab.classification_evaluate import evaluate_classification
from statvocab.classification_features import (
    completed_gold_labels,
    ensure_gold_templates,
    feature_rows,
    write_classification_features,
    write_csv_rows,
    write_parquet_rows,
)
from statvocab.classify_embeddings import generate_term_embeddings
from statvocab.classify_rules import classify_feature_row
from statvocab.config import AppConfig
from statvocab.contracts import ClassificationPrediction, ResourceRecord, VocabularyCategory
from statvocab.manifests import complete_manifest, create_manifest, write_manifest

SUPPORTED_VARIANTS = {"rule-only", "embedding-enhanced", "local-hybrid", "paid-adjudicated"}
CSV_BY_CATEGORY = {
    VocabularyCategory.MEASURE: "measures.csv",
    VocabularyCategory.DIMENSION_NAME: "dimension_names.csv",
    VocabularyCategory.DIMENSION_VALUE: "dimension_values.csv",
    VocabularyCategory.UNIT: "units.csv",
    VocabularyCategory.OTHER_AMBIGUOUS: "other_ambiguous.csv",
}
EXPORT_FIELDNAMES = [
    "term_id",
    "canonical_term",
    "category",
    "confidence",
    "variant",
    "protected",
    "evidence",
    "occurrence_ids_json",
    "run_id",
]


def _occurrence_ids(row: dict[str, Any]) -> tuple[str, ...]:
    value = row.get("occurrence_ids_json") or "[]"
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        return ()
    return tuple(sorted(str(item) for item in parsed))


def _prediction_row(
    feature_row: dict[str, Any],
    *,
    category: VocabularyCategory,
    confidence: float,
    evidence: str,
    protected: bool,
    variant: str,
    run_id: str,
) -> dict[str, Any]:
    prediction = ClassificationPrediction(
        term_id=str(feature_row["term_id"]),
        canonical_term=str(feature_row["canonical_term"]),
        category=category,
        confidence=confidence,
        variant=variant,
        protected=protected,
        evidence=evidence,
        occurrence_ids=_occurrence_ids(feature_row),
        run_id=run_id,
    )
    payload = prediction.model_dump(mode="json")
    payload["occurrence_ids_json"] = json.dumps(payload.pop("occurrence_ids"), sort_keys=True)
    return payload


def _rule_predictions(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    run_id: str,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        decision = classify_feature_row(row)
        predictions.append(
            _prediction_row(
                row,
                category=decision.category,
                confidence=decision.confidence,
                evidence=decision.evidence,
                protected=decision.protected,
                variant=variant,
                run_id=run_id,
            )
        )
    return predictions


def _import_sklearn() -> tuple[Any, Any, Any]:
    try:
        feature_extraction = import_module("sklearn.feature_extraction")
        linear_model = import_module("sklearn.linear_model")
        calibration = import_module("sklearn.calibration")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Local-hybrid classification with completed gold labels requires scikit-learn. "
            "Install with `python -m pip install -e .[ml]`."
        ) from exc
    return (
        feature_extraction.DictVectorizer,
        linear_model.LogisticRegression,
        calibration.CalibratedClassifierCV,
    )


def _model_feature_dict(row: dict[str, Any]) -> dict[str, object]:
    return {
        "source_roles": row["source_roles"],
        "frequency_bin": row["frequency_bin"],
        "table_count_bin": row["table_count_bin"],
        "has_title_evidence": bool(row["has_title_evidence"]),
        "has_header_evidence": bool(row["has_header_evidence"]),
        "has_metadata_value_evidence": bool(row["has_metadata_value_evidence"]),
        "has_conflicting_roles": bool(row["has_conflicting_roles"]),
        "has_percent": bool(row["has_percent"]),
        "has_digit": bool(row["has_digit"]),
        "has_age_pattern": bool(row["has_age_pattern"]),
        "is_short_code": bool(row["is_short_code"]),
        "has_unit_word": bool(row["has_unit_word"]),
        "has_value_word": bool(row["has_value_word"]),
        "looks_like_dimension_name": bool(row["looks_like_dimension_name"]),
        "log_occurrence_count": min(int(row["occurrence_count"]), 1000),
        "log_table_count": min(int(row["table_count"]), 1000),
    }


def _hybrid_predictions(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    run_id: str,
    config: AppConfig,
) -> list[dict[str, Any]]:
    labels = completed_gold_labels(config)
    if not labels:
        fallback_predictions = _rule_predictions(rows, variant=variant, run_id=run_id)
        for prediction in fallback_predictions:
            prediction["evidence"] = (
                prediction["evidence"] + "; local model skipped because gold labels are pending"
            )
        return fallback_predictions

    generate_term_embeddings(config, rows)
    dict_vectorizer, logistic_regression, calibrated_classifier_cv = _import_sklearn()
    row_by_term = {str(row["term_id"]): row for row in rows}
    train_labels = [row for row in labels if row.get("split") == "train_dev"]
    if len({row["category"] for row in train_labels}) < 2:
        return _rule_predictions(rows, variant=variant, run_id=run_id)

    vectorizer = dict_vectorizer(sparse=True)
    train_x = vectorizer.fit_transform(
        [_model_feature_dict(row_by_term[row["term_id"]]) for row in train_labels]
    )
    train_y = [row["category"] for row in train_labels]
    base = logistic_regression(max_iter=1000, random_state=config.random_seed)
    model = calibrated_classifier_cv(base, cv=min(3, len(train_labels)))
    model.fit(train_x, train_y)

    rule_rows = _rule_predictions(rows, variant=variant, run_id=run_id)
    all_x = vectorizer.transform([_model_feature_dict(row) for row in rows])
    probabilities = model.predict_proba(all_x)
    classes = [str(value) for value in model.classes_]
    predictions: list[dict[str, Any]] = []
    for row, rule_row, probs in zip(rows, rule_rows, probabilities, strict=True):
        if bool(rule_row["protected"]):
            predictions.append(rule_row)
            continue
        best_index = max(range(len(classes)), key=lambda index: float(probs[index]))
        confidence = float(probs[best_index])
        if confidence < config.classification.abstention_threshold:
            category = VocabularyCategory.OTHER_AMBIGUOUS
            evidence = "local classifier abstained below validation threshold"
        else:
            category = VocabularyCategory(classes[best_index])
            evidence = "local calibrated classifier prediction"
        predictions.append(
            _prediction_row(
                row,
                category=category,
                confidence=confidence,
                evidence=evidence,
                protected=False,
                variant=variant,
                run_id=run_id,
            )
        )
    return predictions


def _write_predictions(rows: list[dict[str, Any]], output_dir: Path, variant: str) -> Path:
    return write_parquet_rows(rows, output_dir / f"{variant}_predictions.parquet")


def _export_category_csvs(rows: list[dict[str, Any]], outputs_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for category, filename in CSV_BY_CATEGORY.items():
        selected = [
            {
                **row,
                "category": str(row["category"]),
            }
            for row in rows
            if row["category"] == category.value
        ]
        selected.sort(key=lambda row: (str(row["canonical_term"]).casefold(), str(row["term_id"])))
        paths.append(write_csv_rows(selected, outputs_dir / filename, EXPORT_FIELDNAMES))
    return paths


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _classification_summary(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    run_id: str,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        counts[category] = counts.get(category, 0) + 1
    return {
        "run_id": run_id,
        "variant": variant,
        "prediction_count": len(rows),
        "category_counts": dict(sorted(counts.items())),
        "accepted_hallucination_count": 0,
    }


def run_classification(
    config: AppConfig,
    *,
    variant: str,
    resources: tuple[ResourceRecord, ...] = (),
) -> tuple[dict[str, Path], Path, dict[str, Any]]:
    """Run one Milestone 3 classification variant."""

    if variant not in SUPPORTED_VARIANTS:
        allowed = ", ".join(sorted(SUPPORTED_VARIANTS))
        raise ValueError(
            f"Unsupported classification variant {variant!r}; expected one of: {allowed}"
        )
    if variant == "paid-adjudicated":
        raise RuntimeError("Paid adjudication is scaffolded but disabled by default.")

    manifest = create_manifest(config, f"classify-{variant}")
    run_id = manifest.run_id
    features_path = write_classification_features(config)
    sample_path, labels_path, relabel_path = ensure_gold_templates(config)
    rows = feature_rows(config)
    if variant == "rule-only":
        predictions = _rule_predictions(rows, variant=variant, run_id=run_id)
    else:
        predictions = _hybrid_predictions(rows, variant=variant, run_id=run_id, config=config)

    output_dir = config.paths.outputs_dir / "classification" / run_id
    predictions_path = _write_predictions(predictions, output_dir, variant)
    export_paths = _export_category_csvs(predictions, config.paths.outputs_dir)
    metrics_path, metrics_payload = evaluate_classification(
        config,
        prediction_rows=predictions,
        variant=variant,
        output_path=output_dir / "classification_metrics.json",
    )
    summary = _classification_summary(predictions, variant=variant, run_id=run_id)
    summary["metrics_status"] = metrics_payload.get("metrics_status", "available")
    summary_path = _write_json(summary, output_dir / "classification_summary.json")

    artifacts = {
        "features": features_path,
        "gold_sample": sample_path,
        "gold_labels": labels_path,
        "gold_relabel": relabel_path,
        "predictions": predictions_path,
        "metrics": metrics_path,
        "summary": summary_path,
    }
    for path in export_paths:
        artifacts[path.stem] = path
    completed = complete_manifest(
        manifest,
        resources=tuple(record.resource_id for record in resources),
        artifacts=tuple(str(path) for path in artifacts.values()),
    )
    manifest_path = write_manifest(
        completed,
        config.paths.outputs_dir / "manifests" / f"{manifest.run_id}_classify_{variant}.json",
    )
    return artifacts, manifest_path, summary
