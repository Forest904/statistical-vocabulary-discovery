"""Milestone 3 semantic partition runner."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from statvocab.classification_evaluate import _score_rows, evaluate_classification
from statvocab.classification_features import (
    completed_gold_labels,
    ensure_gold_templates,
    feature_rows,
    read_csv_rows,
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = completed_gold_labels(config)
    if not labels:
        fallback_predictions = _rule_predictions(rows, variant=variant, run_id=run_id)
        for prediction in fallback_predictions:
            prediction["evidence"] = (
                prediction["evidence"] + "; local model skipped because gold labels are pending"
            )
        return fallback_predictions, {"threshold": config.classification.abstention_threshold}

    generate_term_embeddings(config, rows)
    dict_vectorizer, logistic_regression, calibrated_classifier_cv = _import_sklearn()
    row_by_term = {str(row["term_id"]): row for row in rows}
    train_labels = [row for row in labels if row.get("split") == "train_dev"]
    if len({row["category"] for row in train_labels}) < 2:
        return _rule_predictions(rows, variant=variant, run_id=run_id), {
            "threshold": config.classification.abstention_threshold,
            "message": "local model skipped because train labels have fewer than two classes",
        }

    vectorizer = dict_vectorizer(sparse=True)
    train_x = vectorizer.fit_transform(
        [_model_feature_dict(row_by_term[row["term_id"]]) for row in train_labels]
    )
    train_y = [row["category"] for row in train_labels]
    base = logistic_regression(
        max_iter=config.classification.classifier_max_iter,
        random_state=config.random_seed,
        class_weight="balanced" if config.classification.balance_class_weight else None,
    )
    model = calibrated_classifier_cv(base, cv=min(3, len(train_labels)))
    model.fit(train_x, train_y)

    rule_rows = _rule_predictions(rows, variant=variant, run_id=run_id)
    all_x = vectorizer.transform([_model_feature_dict(row) for row in rows])
    probabilities = model.predict_proba(all_x)
    classes = [str(value) for value in model.classes_]

    def build_predictions(non_other_threshold: float) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        for row, rule_row, probs in zip(rows, rule_rows, probabilities, strict=True):
            if bool(rule_row["protected"]):
                predictions.append(rule_row)
                continue
            best_index = max(range(len(classes)), key=lambda index: float(probs[index]))
            confidence = float(probs[best_index])
            best_class = classes[best_index]
            if best_class != VocabularyCategory.OTHER_AMBIGUOUS.value:
                if confidence < non_other_threshold:
                    category = VocabularyCategory.OTHER_AMBIGUOUS
                    evidence = "local classifier abstained below validation threshold"
                else:
                    category = VocabularyCategory(best_class)
                    evidence = "local calibrated classifier prediction"
            elif confidence < config.classification.abstention_threshold:
                category = VocabularyCategory.OTHER_AMBIGUOUS
                evidence = "local classifier abstained below validation threshold"
            else:
                category = VocabularyCategory.OTHER_AMBIGUOUS
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

    validation_labels = [row for row in labels if row.get("split") == "validation"]
    threshold_candidates = [round(0.45 + (index * 0.025), 3) for index in range(13)]
    selected_threshold = config.classification.abstention_threshold
    selected_payload: dict[str, Any] = {"reason": "default threshold used"}
    best_score = -1.0
    for threshold in threshold_candidates:
        trial_predictions = build_predictions(threshold)
        if not validation_labels:
            continue
        scored = _score_rows(validation_labels, trial_predictions)
        non_other_precision = _non_other_precision(validation_labels, trial_predictions)
        if (
            non_other_precision is not None
            and non_other_precision < config.classification.non_other_precision_floor
        ):
            continue
        score = float(scored["macro_f1"])
        if score > best_score:
            best_score = score
            selected_threshold = threshold
            selected_payload = {
                "reason": "selected on validation macro-F1 with non-other precision floor",
                "validation_macro_f1": score,
                "validation_non_other_precision": non_other_precision,
            }
    predictions = build_predictions(selected_threshold)
    selected_payload["threshold"] = selected_threshold
    return predictions, selected_payload


def _non_other_precision(
    gold_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, Any]],
) -> float | None:
    by_term = {str(row["term_id"]): str(row["category"]) for row in prediction_rows}
    predicted = [
        row
        for row in gold_rows
        if by_term.get(row["term_id"]) not in {None, VocabularyCategory.OTHER_AMBIGUOUS.value}
    ]
    if not predicted:
        return None
    return sum(1 for row in predicted if by_term[row["term_id"]] == row["category"]) / len(
        predicted
    )


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


def _read_existing_exports(outputs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, filename in CSV_BY_CATEGORY.items():
        for row in read_csv_rows(outputs_dir / filename):
            row["category"] = category.value
            rows.append(row)
    return rows


def _category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _gate_exports(
    *,
    config: AppConfig,
    existing_predictions: list[dict[str, Any]],
    candidate_predictions: list[dict[str, Any]],
    metrics_payload: dict[str, Any],
) -> dict[str, Any]:
    labels = completed_gold_labels(config)
    random_final = [
        row
        for row in labels
        if (row.get("audit_source") or "random_sample") == "random_sample"
        and row.get("split") == "final_test"
    ]
    targeted_final = [
        row
        for row in labels
        if (row.get("audit_source") or "") == "targeted_reclaim"
        and row.get("split") == "final_test"
    ]
    reasons: list[str] = []
    if not existing_predictions:
        return {"accepted": True, "reasons": ["no existing submitted exports to protect"]}
    if not targeted_final:
        return {
            "accepted": False,
            "reasons": ["targeted reclaim final-test labels are pending"],
        }

    existing_random = _score_rows(random_final, existing_predictions)
    candidate_random = _score_rows(random_final, candidate_predictions)
    macro_drop = float(existing_random["macro_f1"]) - float(candidate_random["macro_f1"])
    if macro_drop > config.classification.acceptance_final_macro_drop:
        reasons.append(
            "random final-test macro-F1 drop "
            f"{macro_drop:.3f} exceeds {config.classification.acceptance_final_macro_drop:.3f}"
        )

    targeted_precision = _non_other_precision(targeted_final, candidate_predictions)
    if (
        targeted_precision is None
        or targeted_precision < config.classification.non_other_precision_floor
    ):
        reasons.append(
            "targeted final-test non-other precision "
            f"{targeted_precision if targeted_precision is not None else 'n/a'} below "
            f"{config.classification.non_other_precision_floor:.2f}"
        )

    existing_counts = _category_counts(existing_predictions)
    candidate_counts = _category_counts(candidate_predictions)
    existing_other = existing_counts.get(VocabularyCategory.OTHER_AMBIGUOUS.value, 0)
    candidate_other = candidate_counts.get(VocabularyCategory.OTHER_AMBIGUOUS.value, 0)
    reduction = ((existing_other - candidate_other) / existing_other) if existing_other else 1.0
    if reduction < config.classification.min_other_reduction_fraction:
        reasons.append(
            "other_ambiguous reduction "
            f"{reduction:.3f} below {config.classification.min_other_reduction_fraction:.3f}"
        )

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "baseline_random_final_macro_f1": existing_random["macro_f1"],
        "candidate_random_final_macro_f1": candidate_random["macro_f1"],
        "targeted_final_non_other_precision": targeted_precision,
        "baseline_other_ambiguous": existing_other,
        "candidate_other_ambiguous": candidate_other,
        "other_reduction_fraction": reduction,
        "source_metrics_available": bool(metrics_payload.get("source_metrics")),
    }


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
        model_selection = {"threshold": None, "reason": "rule-only classifier"}
    else:
        predictions, model_selection = _hybrid_predictions(
            rows,
            variant=variant,
            run_id=run_id,
            config=config,
        )

    output_dir = config.paths.outputs_dir / "classification" / run_id
    predictions_path = _write_predictions(predictions, output_dir, variant)
    metrics_path, metrics_payload = evaluate_classification(
        config,
        prediction_rows=predictions,
        variant=variant,
        output_path=output_dir / "classification_metrics.json",
    )
    existing_exports = _read_existing_exports(config.paths.outputs_dir)
    gate = (
        _gate_exports(
            config=config,
            existing_predictions=existing_exports,
            candidate_predictions=predictions,
            metrics_payload=metrics_payload,
        )
        if variant == "local-hybrid"
        else {"accepted": True, "reasons": ["gate not applied to this variant"]}
    )
    export_dir = config.paths.outputs_dir if bool(gate["accepted"]) else output_dir / "proposed"
    export_paths = _export_category_csvs(predictions, export_dir)
    summary = _classification_summary(predictions, variant=variant, run_id=run_id)
    summary["metrics_status"] = metrics_payload.get("metrics_status", "available")
    summary["model_selection"] = model_selection
    summary["acceptance_gate"] = gate
    summary["exports_promoted"] = bool(gate["accepted"])
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
