"""Milestone 3 semantic partition runner."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from statvocab.classification_evaluate import _score_rows, evaluate_classification
from statvocab.classification_features import (
    completed_gold_labels,
    ensure_gold_templates,
    feature_rows,
    read_csv_rows,
    read_parquet_rows,
    write_classification_features,
    write_csv_rows,
    write_parquet_rows,
)
from statvocab.classify_embeddings import generate_term_embeddings
from statvocab.classify_rules import classify_feature_row
from statvocab.config import AppConfig
from statvocab.contracts import ClassificationPrediction, ResourceRecord, VocabularyCategory
from statvocab.manifests import complete_manifest, create_manifest, write_manifest

SUPPORTED_VARIANTS = {
    "rule-only",
    "embedding-enhanced",
    "local-hybrid",
    "semantic-hybrid",
    "paid-adjudicated",
}
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
SEMANTIC_OVERRIDE_MODES = ("centroid-and-neighbor", "neighbor-only", "centroid-only")


@dataclass(frozen=True)
class _SemanticOverrideSettings:
    mode: str = "centroid-and-neighbor"
    min_centroid_score: float = 0.72
    min_margin: float = 0.015
    min_neighbor_vote: float = 0.60


@dataclass(frozen=True)
class _CalibrationSettings:
    non_other_threshold: float
    measure_threshold: float
    override: _SemanticOverrideSettings


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


def _model_feature_dict(
    row: dict[str, Any],
    semantic_features: dict[str, object] | None = None,
) -> dict[str, object]:
    features: dict[str, object] = {
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
    if semantic_features:
        features.update(semantic_features)
    return features


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _normalized(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    size = len(vectors[0])
    return _normalized(
        [sum(vector[index] for vector in vectors) / len(vectors) for index in range(size)]
    )


def _load_term_embeddings(config: AppConfig) -> dict[str, list[float]]:
    path = config.paths.processed_dir / "term_embeddings.parquet"
    if not path.exists():
        return {}
    rows = read_parquet_rows(path)
    return {
        str(row["term_id"]): _normalized(
            [float(value) for value in cast(list[Any], row["embedding"])]
        )
        for row in rows
        if row.get("model") == config.classification.pearl_model
        and row.get("revision") == config.classification.pearl_revision
    }


def semantic_feature_rows(
    config: AppConfig,
    rows: list[dict[str, Any]],
    labels: list[dict[str, str]],
    embeddings: dict[str, list[float]] | None = None,
) -> dict[str, dict[str, object]]:
    """Build interpretable semantic features from labeled embedding neighborhoods."""

    vectors_by_term = embeddings if embeddings is not None else _load_term_embeddings(config)
    train_labels = [row for row in labels if row.get("split") == "train_dev"]
    labeled: list[tuple[str, str, list[float]]] = []
    by_category: dict[str, list[list[float]]] = {
        category.value: [] for category in VocabularyCategory
    }
    for label in train_labels:
        term_id = str(label["term_id"])
        category = str(label["category"])
        vector = vectors_by_term.get(term_id)
        if vector is None:
            continue
        labeled.append((term_id, category, vector))
        by_category.setdefault(category, []).append(vector)

    centroids = {
        category: _centroid(vectors)
        for category, vectors in by_category.items()
        if vectors
    }
    category_values = [category.value for category in VocabularyCategory]
    features_by_term: dict[str, dict[str, object]] = {}
    for row in rows:
        term_id = str(row["term_id"])
        vector = vectors_by_term.get(term_id)
        features: dict[str, object] = {
            "semantic_best_centroid_class": "none",
            "semantic_best_centroid_score": 0.0,
            "semantic_centroid_margin": 0.0,
            "semantic_neighbor_best_class": "none",
            "semantic_neighbor_best_vote": 0.0,
        }
        for category in category_values:
            features[f"semantic_centroid_{category}"] = 0.0
            features[f"semantic_neighbor_vote_{category}"] = 0.0
            features[f"semantic_neighbor_confidence_{category}"] = 0.0
        if vector is None:
            features_by_term[term_id] = features
            continue

        centroid_scores = {
            category: _dot(vector, centroid)
            for category, centroid in centroids.items()
            if centroid
        }
        for category, score in centroid_scores.items():
            features[f"semantic_centroid_{category}"] = score
        ordered_centroids = sorted(
            centroid_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if ordered_centroids:
            best_category, best_score = ordered_centroids[0]
            second_score = ordered_centroids[1][1] if len(ordered_centroids) > 1 else 0.0
            features["semantic_best_centroid_class"] = best_category
            features["semantic_best_centroid_score"] = best_score
            features["semantic_centroid_margin"] = best_score - second_score

        neighbors = sorted(
            (
                (category, _dot(vector, label_vector))
                for label_term_id, category, label_vector in labeled
                if label_term_id != term_id
            ),
            key=lambda item: item[1],
            reverse=True,
        )[: config.classification.semantic_neighbor_k]
        if neighbors:
            counts: dict[str, int] = {}
            confidence_sum: dict[str, float] = {}
            for category, score in neighbors:
                counts[category] = counts.get(category, 0) + 1
                confidence_sum[category] = confidence_sum.get(category, 0.0) + max(score, 0.0)
            best_neighbor_category, best_count = sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]
            features["semantic_neighbor_best_class"] = best_neighbor_category
            features["semantic_neighbor_best_vote"] = best_count / len(neighbors)
            for category in category_values:
                features[f"semantic_neighbor_vote_{category}"] = counts.get(category, 0) / len(
                    neighbors
                )
                features[f"semantic_neighbor_confidence_{category}"] = confidence_sum.get(
                    category,
                    0.0,
                ) / len(neighbors)
        features_by_term[term_id] = features
    return features_by_term


def _recall_bonus(scored: dict[str, Any]) -> float:
    per_class = cast(dict[str, dict[str, float | int]], scored.get("per_class") or {})
    measure = float(per_class.get(VocabularyCategory.MEASURE.value, {}).get("recall", 0.0))
    dimension_value = float(
        per_class.get(VocabularyCategory.DIMENSION_VALUE.value, {}).get("recall", 0.0)
    )
    unit = float(per_class.get(VocabularyCategory.UNIT.value, {}).get("recall", 0.0))
    return (measure + dimension_value + unit) / 3.0


def _selection_score(config: AppConfig, scored: dict[str, Any]) -> float:
    macro_f1 = float(scored["macro_f1"])
    if config.classification.threshold_objective == "macro_f1_with_recall_bonus":
        return macro_f1 + (0.25 * _recall_bonus(scored))
    return macro_f1


def _threshold_values(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 3))
        current += step
    return values


def _semantic_calibration_grid() -> list[_CalibrationSettings]:
    settings: list[_CalibrationSettings] = []
    for non_other_threshold in _threshold_values(0.55, 0.90, 0.025):
        for measure_threshold in _threshold_values(0.60, 0.95, 0.025):
            for min_centroid_score in (0.72, 0.76, 0.80, 0.84):
                for min_margin in (0.015, 0.03, 0.05, 0.075, 0.10):
                    for min_neighbor_vote in (0.60, 0.75, 1.00):
                        for mode in SEMANTIC_OVERRIDE_MODES:
                            settings.append(
                                _CalibrationSettings(
                                    non_other_threshold=non_other_threshold,
                                    measure_threshold=measure_threshold,
                                    override=_SemanticOverrideSettings(
                                        mode=mode,
                                        min_centroid_score=min_centroid_score,
                                        min_margin=min_margin,
                                        min_neighbor_vote=min_neighbor_vote,
                                    ),
                                )
                            )
    return settings


def _settings_payload(settings: _CalibrationSettings) -> dict[str, Any]:
    return {
        "non_other_threshold": settings.non_other_threshold,
        "measure_threshold": settings.measure_threshold,
        "semantic_override_mode": settings.override.mode,
        "semantic_override_min_centroid_score": settings.override.min_centroid_score,
        "semantic_override_min_margin": settings.override.min_margin,
        "semantic_override_min_neighbor_vote": settings.override.min_neighbor_vote,
    }


def _override_mode_rank(mode: str) -> int:
    return {
        "centroid-and-neighbor": 2,
        "neighbor-only": 1,
        "centroid-only": 0,
    }.get(mode, 0)


def _semantic_measure_override(
    semantic_features: dict[str, object] | None,
    settings: _SemanticOverrideSettings | None = None,
) -> float | None:
    if not semantic_features:
        return None
    actual = settings or _SemanticOverrideSettings()
    best_class = str(semantic_features.get("semantic_best_centroid_class") or "")
    neighbor_class = str(semantic_features.get("semantic_neighbor_best_class") or "")
    measure_score = float(semantic_features.get("semantic_centroid_measure") or 0.0)
    other_score = float(semantic_features.get("semantic_centroid_other_ambiguous") or 0.0)
    dimension_score = float(semantic_features.get("semantic_centroid_dimension_value") or 0.0)
    neighbor_vote = float(semantic_features.get("semantic_neighbor_vote_measure") or 0.0)
    margin = measure_score - max(other_score, dimension_score)
    centroid_pass = (
        best_class == VocabularyCategory.MEASURE.value
        and measure_score >= actual.min_centroid_score
        and margin >= actual.min_margin
        and measure_score > dimension_score
    )
    neighbor_pass = (
        neighbor_class == VocabularyCategory.MEASURE.value
        and neighbor_vote >= actual.min_neighbor_vote
        and measure_score >= actual.min_centroid_score
        and measure_score > dimension_score
    )
    if actual.mode == "centroid-only" and centroid_pass:
        return min(0.95, measure_score)
    if actual.mode == "neighbor-only" and neighbor_pass:
        return min(0.92, max(measure_score, neighbor_vote))
    if actual.mode == "centroid-and-neighbor" and centroid_pass and neighbor_pass:
        return min(0.95, max(measure_score, neighbor_vote))
    return None


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

    embedding_started = time.perf_counter()
    generate_term_embeddings(config, rows)
    embedding_seconds = time.perf_counter() - embedding_started
    semantic_started = time.perf_counter()
    semantic_features = (
        semantic_feature_rows(config, rows, labels)
        if variant == "semantic-hybrid" and config.classification.semantic_hybrid_enabled
        else {}
    )
    semantic_feature_seconds = time.perf_counter() - semantic_started
    dict_vectorizer, logistic_regression, calibrated_classifier_cv = _import_sklearn()
    row_by_term = {str(row["term_id"]): row for row in rows}
    row_index_by_term = {str(row["term_id"]): index for index, row in enumerate(rows)}
    train_labels = [row for row in labels if row.get("split") == "train_dev"]
    if len({row["category"] for row in train_labels}) < 2:
        return _rule_predictions(rows, variant=variant, run_id=run_id), {
            "threshold": config.classification.abstention_threshold,
            "message": "local model skipped because train labels have fewer than two classes",
        }

    vectorizer = dict_vectorizer(sparse=True)
    train_x = vectorizer.fit_transform(
        [
            _model_feature_dict(
                row_by_term[row["term_id"]],
                semantic_features.get(str(row["term_id"])),
            )
            for row in train_labels
        ]
    )
    train_y = [row["category"] for row in train_labels]
    base = logistic_regression(
        max_iter=config.classification.classifier_max_iter,
        random_state=config.random_seed,
        class_weight="balanced" if config.classification.balance_class_weight else None,
    )
    fit_started = time.perf_counter()
    model = calibrated_classifier_cv(base, cv=min(3, len(train_labels)))
    model.fit(train_x, train_y)
    model_fit_seconds = time.perf_counter() - fit_started

    rule_rows = _rule_predictions(rows, variant=variant, run_id=run_id)
    predict_started = time.perf_counter()
    all_x = vectorizer.transform(
        [_model_feature_dict(row, semantic_features.get(str(row["term_id"]))) for row in rows]
    )
    probabilities = model.predict_proba(all_x)
    classes = [str(value) for value in model.classes_]
    probability_seconds = time.perf_counter() - predict_started

    def build_predictions(
        settings: _CalibrationSettings,
        *,
        indexes: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        actual_indexes = indexes if indexes is not None else list(range(len(rows)))
        predictions: list[dict[str, Any]] = []
        for row_index in actual_indexes:
            row = rows[row_index]
            rule_row = rule_rows[row_index]
            probs = probabilities[row_index]
            if bool(rule_row["protected"]):
                predictions.append(rule_row)
                continue
            row_semantic_features = semantic_features.get(str(row["term_id"]))
            best_index = max(range(len(classes)), key=lambda index: float(probs[index]))
            confidence = float(probs[best_index])
            best_class = classes[best_index]
            semantic_measure_confidence = (
                _semantic_measure_override(row_semantic_features, settings.override)
                if variant == "semantic-hybrid"
                else None
            )
            if (
                semantic_measure_confidence is not None
                and semantic_measure_confidence >= settings.measure_threshold
            ):
                category = VocabularyCategory.MEASURE
                confidence = max(confidence, semantic_measure_confidence)
                evidence = "semantic centroid/neighbor measure fallback"
            elif best_class != VocabularyCategory.OTHER_AMBIGUOUS.value:
                threshold = (
                    settings.measure_threshold
                    if best_class == VocabularyCategory.MEASURE.value
                    else settings.non_other_threshold
                )
                if confidence < threshold:
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
    validation_indexes = [
        row_index_by_term[str(label["term_id"])]
        for label in validation_labels
        if str(label["term_id"]) in row_index_by_term
    ]
    settings_candidates = (
        _semantic_calibration_grid()
        if variant == "semantic-hybrid"
        else [
            _CalibrationSettings(
                non_other_threshold=threshold,
                measure_threshold=threshold,
                override=_SemanticOverrideSettings(),
            )
            for threshold in [round(0.45 + (index * 0.025), 3) for index in range(13)]
        ]
    )
    selected_settings = _CalibrationSettings(
        non_other_threshold=config.classification.abstention_threshold,
        measure_threshold=config.classification.abstention_threshold,
        override=_SemanticOverrideSettings(),
    )
    selected_payload: dict[str, Any] = {"reason": "default threshold used"}
    best_rank: tuple[bool, bool, float, float, float, int] | None = None
    tradeoff_rows: list[dict[str, Any]] = []
    sweep_started = time.perf_counter()
    for settings in settings_candidates:
        if not validation_labels:
            continue
        trial_predictions = build_predictions(settings, indexes=validation_indexes)
        scored = _score_rows(validation_labels, trial_predictions)
        non_other_precision = _non_other_precision(validation_labels, trial_predictions)
        gate_eligible = not (
            non_other_precision is not None
            and non_other_precision < config.classification.non_other_precision_floor
        )
        per_class = cast(dict[str, dict[str, float | int]], scored.get("per_class") or {})
        measure_recall = float(
            per_class.get(VocabularyCategory.MEASURE.value, {}).get("recall", 0.0)
        )
        promotion_proxy_eligible = (
            gate_eligible
            and measure_recall >= config.classification.min_measure_recall_for_promotion
        )
        score = _selection_score(config, scored)
        tradeoff_rows.append(
            {
                **_settings_payload(settings),
                "validation_macro_f1": float(scored["macro_f1"]),
                "validation_objective_score": score,
                "validation_non_other_precision": non_other_precision,
                "validation_measure_recall": measure_recall,
                "gate_eligible": gate_eligible,
                "promotion_proxy_eligible": promotion_proxy_eligible,
            }
        )
        if not gate_eligible:
            continue
        rank = (
            promotion_proxy_eligible,
            gate_eligible,
            float(non_other_precision or 0.0),
            _override_mode_rank(settings.override.mode),
            score,
            measure_recall,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            selected_settings = settings
            selected_payload = {
                "reason": (
                    "selected on validation promotion proxy"
                    if promotion_proxy_eligible
                    else "selected on validation macro-F1 with non-other precision floor"
                ),
                "validation_macro_f1": float(scored["macro_f1"]),
                "validation_objective_score": score,
                "validation_non_other_precision": non_other_precision,
                "validation_measure_recall": measure_recall,
                "validation_promotion_proxy_eligible": promotion_proxy_eligible,
            }
    sweep_seconds = time.perf_counter() - sweep_started
    if best_rank is None and validation_labels:
        selected_payload = {
            "reason": "no validation candidate satisfied non-other precision floor; default used"
        }
    full_predict_started = time.perf_counter()
    predictions = build_predictions(selected_settings)
    full_prediction_seconds = time.perf_counter() - full_predict_started
    selected_payload.update(_settings_payload(selected_settings))
    selected_payload["threshold"] = selected_settings.non_other_threshold
    selected_payload["semantic_features_enabled"] = bool(semantic_features)
    selected_payload["timing"] = {
        "embedding_seconds": embedding_seconds,
        "semantic_feature_seconds": semantic_feature_seconds,
        "model_fit_seconds": model_fit_seconds,
        "probability_seconds": probability_seconds,
        "calibration_sweep_seconds": sweep_seconds,
        "full_prediction_seconds": full_prediction_seconds,
    }
    if variant == "semantic-hybrid":
        selected_payload["calibration"] = {
            "candidate_count": len(settings_candidates),
            "evaluated_candidate_count": len(tradeoff_rows),
            "selected_settings": _settings_payload(selected_settings),
            "tradeoff_table": sorted(
                tradeoff_rows,
                key=lambda row: (
                    not bool(row["promotion_proxy_eligible"]),
                    not bool(row["gate_eligible"]),
                    -float(row["validation_objective_score"]),
                    -float(row.get("validation_non_other_precision") or 0.0),
                ),
            )[:500],
            "tradeoff_table_truncated": len(tradeoff_rows) > 500,
        }
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
    variant: str,
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
    existing_ids = {str(row["term_id"]) for row in existing_predictions}
    candidate_ids = {str(row["term_id"]) for row in candidate_predictions}
    if existing_ids != candidate_ids:
        missing = len(candidate_ids - existing_ids)
        extra = len(existing_ids - candidate_ids)
        reason = (
            "existing submitted exports are incompatible with current vocabulary "
            f"({missing} missing, {extra} extra term IDs)"
        )
        if variant == "local-hybrid":
            return {
                "accepted": True,
                "reasons": [reason, "local-hybrid bootstrap accepted"],
                "baseline_compatible": False,
            }
        return {
            "accepted": False,
            "reasons": [reason, "run local-hybrid before promoting semantic-hybrid"],
            "baseline_compatible": False,
        }
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
        or targeted_precision < config.classification.min_non_other_precision
    ):
        reasons.append(
            "targeted final-test non-other precision "
            f"{targeted_precision if targeted_precision is not None else 'n/a'} below "
            f"{config.classification.min_non_other_precision:.2f}"
        )

    existing_targeted = _score_rows(targeted_final, existing_predictions)
    candidate_targeted = _score_rows(targeted_final, candidate_predictions)
    if float(candidate_targeted["macro_f1"]) <= float(existing_targeted["macro_f1"]):
        reasons.append(
            "targeted reclaim macro-F1 did not improve "
            f"baseline={existing_targeted['macro_f1']:.3f} "
            f"candidate={candidate_targeted['macro_f1']:.3f}"
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

    candidate_per_class = cast(dict[str, dict[str, float | int]], candidate_targeted["per_class"])
    candidate_measure_recall = float(
        candidate_per_class.get(VocabularyCategory.MEASURE.value, {}).get("recall", 0.0)
    )
    existing_measure_recall = float(
        cast(dict[str, dict[str, float | int]], existing_targeted["per_class"])
        .get(VocabularyCategory.MEASURE.value, {})
        .get("recall", 0.0)
    )
    if candidate_measure_recall < config.classification.min_measure_recall_for_promotion:
        reasons.append(
            "targeted measure recall "
            f"{candidate_measure_recall:.3f} below "
            f"{config.classification.min_measure_recall_for_promotion:.3f}"
        )
    if candidate_measure_recall <= existing_measure_recall:
        reasons.append(
            "targeted measure recall did not improve "
            f"baseline={existing_measure_recall:.3f} candidate={candidate_measure_recall:.3f}"
        )

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "baseline_random_final_macro_f1": existing_random["macro_f1"],
        "candidate_random_final_macro_f1": candidate_random["macro_f1"],
        "targeted_final_non_other_precision": targeted_precision,
        "baseline_targeted_final_macro_f1": existing_targeted["macro_f1"],
        "candidate_targeted_final_macro_f1": candidate_targeted["macro_f1"],
        "baseline_targeted_measure_recall": existing_measure_recall,
        "candidate_targeted_measure_recall": candidate_measure_recall,
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


def _semantic_false_positive_breakdown(
    labels: list[dict[str, str]],
    predictions: list[dict[str, Any]],
) -> dict[str, int]:
    labels_by_term = {str(row["term_id"]): str(row["category"]) for row in labels}
    counts: dict[str, int] = {}
    for prediction in predictions:
        if prediction.get("category") != VocabularyCategory.MEASURE.value:
            continue
        if "semantic centroid/neighbor measure fallback" not in str(prediction.get("evidence")):
            continue
        gold = labels_by_term.get(str(prediction["term_id"]))
        if gold is None or gold == VocabularyCategory.MEASURE.value:
            continue
        counts[gold] = counts.get(gold, 0) + 1
    return dict(sorted(counts.items()))


def _calibration_report_payload(
    *,
    config: AppConfig,
    run_id: str,
    model_selection: dict[str, Any],
    predictions: list[dict[str, Any]],
    metrics_payload: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    calibration = cast(dict[str, Any], model_selection.get("calibration") or {})
    return {
        "run_id": run_id,
        "variant": "semantic-hybrid",
        "config_name": config.config_name,
        "corpus": config.corpus.name,
        "chosen_settings": calibration.get("selected_settings") or {
            key: model_selection.get(key)
            for key in (
                "non_other_threshold",
                "measure_threshold",
                "semantic_override_mode",
                "semantic_override_min_centroid_score",
                "semantic_override_min_margin",
                "semantic_override_min_neighbor_vote",
            )
        },
        "validation_metrics": {
            "macro_f1": model_selection.get("validation_macro_f1"),
            "objective_score": model_selection.get("validation_objective_score"),
            "non_other_precision": model_selection.get("validation_non_other_precision"),
            "measure_recall": model_selection.get("validation_measure_recall"),
        },
        "final_test_promotion_gate": gate,
        "precision_recall_tradeoff_table": calibration.get("tradeoff_table") or [],
        "tradeoff_table_truncated": bool(calibration.get("tradeoff_table_truncated")),
        "evaluated_candidate_count": calibration.get("evaluated_candidate_count", 0),
        "candidate_count": calibration.get("candidate_count", 0),
        "false_positive_category_breakdown": _semantic_false_positive_breakdown(
            completed_gold_labels(config),
            predictions,
        ),
        "timing": model_selection.get("timing") or {},
        "metrics_status": metrics_payload.get("metrics_status", "available"),
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
            variant=variant,
            existing_predictions=existing_exports,
            candidate_predictions=predictions,
            metrics_payload=metrics_payload,
        )
        if variant in {"local-hybrid", "semantic-hybrid"}
        else {"accepted": True, "reasons": ["gate not applied to this variant"]}
    )
    export_dir = config.paths.outputs_dir if bool(gate["accepted"]) else output_dir / "proposed"
    export_paths = _export_category_csvs(predictions, export_dir)
    summary = _classification_summary(predictions, variant=variant, run_id=run_id)
    summary["metrics_status"] = metrics_payload.get("metrics_status", "available")
    summary["model_selection"] = model_selection
    summary["acceptance_gate"] = gate
    summary["exports_promoted"] = bool(gate["accepted"])
    if "timing" in model_selection:
        summary["timing"] = model_selection["timing"]
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
    if variant == "semantic-hybrid":
        calibration_payload = _calibration_report_payload(
            config=config,
            run_id=run_id,
            model_selection=model_selection,
            predictions=predictions,
            metrics_payload=metrics_payload,
            gate=gate,
        )
        artifacts["calibration_report"] = _write_json(
            calibration_payload,
            output_dir / "classification_calibration_metrics.json",
        )
        artifacts["calibration_report_current"] = _write_json(
            calibration_payload,
            config.paths.reports_dir / "classification_calibration_metrics.json",
        )
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
