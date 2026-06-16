import pytest
from pydantic import ValidationError

from statvocab.classification import (
    _CalibrationSettings,
    _evidence_gated_semantic_measure_override,
    _EvidenceGateSettings,
    _gate_exports,
    _semantic_measure_override,
    _SemanticOverrideSettings,
    _settings_payload,
    semantic_feature_rows,
)
from statvocab.classification_context import (
    build_measure_context_features,
    measure_subtype_for_context,
    weak_supervision_vote_summary,
)
from statvocab.classification_evaluate import evaluate_classification
from statvocab.classification_features import write_parquet_rows
from statvocab.classify_embeddings import generate_term_embeddings
from statvocab.classify_llm import validate_llm_response
from statvocab.config import load_config
from statvocab.contracts import ClassificationPrediction, VocabularyCategory


def test_classification_prediction_rejects_duplicate_occurrences() -> None:
    with pytest.raises(ValidationError, match="occurrence_ids must be unique"):
        ClassificationPrediction(
            term_id="term_a",
            canonical_term="Annual",
            category=VocabularyCategory.DIMENSION_VALUE,
            confidence=0.9,
            variant="rule-only",
            evidence="test evidence",
            occurrence_ids=("occ_a", "occ_a"),
            run_id="run_a",
        )


def test_llm_validation_accepts_grounded_response() -> None:
    response = validate_llm_response(
        {
            "term_id": "term_a",
            "category": "unit",
            "confidence": 0.82,
            "evidence_ids": ["occ_a"],
            "rationale": "The term is a unit expression.",
        },
        allowed_term_ids={"term_a"},
        allowed_evidence_ids_by_term={"term_a": {"occ_a"}},
    )

    assert response.category == VocabularyCategory.UNIT


def test_llm_validation_rejects_unknown_term() -> None:
    with pytest.raises(ValueError, match="Unknown term_id"):
        validate_llm_response(
            {
                "term_id": "term_new",
                "category": "measure",
                "confidence": 0.7,
                "evidence_ids": [],
                "rationale": "Invented term.",
            },
            allowed_term_ids={"term_a"},
            allowed_evidence_ids_by_term={"term_a": {"occ_a"}},
        )


def test_llm_validation_rejects_invalid_category() -> None:
    with pytest.raises(ValidationError):
        validate_llm_response(
            {
                "term_id": "term_a",
                "category": "invented",
                "confidence": 0.7,
                "evidence_ids": [],
                "rationale": "Bad category.",
            },
            allowed_term_ids={"term_a"},
            allowed_evidence_ids_by_term={"term_a": {"occ_a"}},
        )


def test_llm_validation_rejects_evidence_from_other_term() -> None:
    with pytest.raises(ValueError, match="outside the term"):
        validate_llm_response(
            {
                "term_id": "term_a",
                "category": "measure",
                "confidence": 0.7,
                "evidence_ids": ["occ_other"],
                "rationale": "Wrong evidence.",
            },
            allowed_term_ids={"term_a"},
            allowed_evidence_ids_by_term={"term_a": {"occ_a"}},
        )


def test_semantic_feature_rows_compute_centroids_and_neighbor_votes() -> None:
    config = load_config("configs/evaluation.yaml")
    rows = [
        {"term_id": "term_measure", "canonical_term": "population"},
        {"term_id": "term_unit", "canonical_term": "euro"},
        {"term_id": "term_candidate", "canonical_term": "urban population"},
    ]
    labels = [
        {"term_id": "term_measure", "category": "measure", "split": "train_dev"},
        {"term_id": "term_unit", "category": "unit", "split": "train_dev"},
    ]
    features = semantic_feature_rows(
        config,
        rows,
        labels,
        embeddings={
            "term_measure": [1.0, 0.0],
            "term_unit": [0.0, 1.0],
            "term_candidate": [0.9, 0.1],
        },
    )

    candidate = features["term_candidate"]
    assert candidate["semantic_best_centroid_class"] == "measure"
    assert float(candidate["semantic_centroid_measure"]) > float(
        candidate["semantic_centroid_unit"]
    )
    assert candidate["semantic_neighbor_best_class"] == "measure"


def test_semantic_measure_override_requires_margin_and_neighbor_agreement() -> None:
    features = {
        "semantic_best_centroid_class": "measure",
        "semantic_neighbor_best_class": "measure",
        "semantic_centroid_measure": 0.82,
        "semantic_centroid_other_ambiguous": 0.60,
        "semantic_centroid_dimension_value": 0.70,
        "semantic_neighbor_vote_measure": 0.75,
    }

    assert _semantic_measure_override(
        features,
        _SemanticOverrideSettings(
            mode="centroid-and-neighbor",
            min_centroid_score=0.80,
            min_margin=0.10,
            min_neighbor_vote=0.75,
        ),
    )
    assert (
        _semantic_measure_override(
            {**features, "semantic_centroid_dimension_value": 0.81},
            _SemanticOverrideSettings(
                mode="centroid-and-neighbor",
                min_centroid_score=0.80,
                min_margin=0.10,
                min_neighbor_vote=0.75,
            ),
        )
        is None
    )
    assert (
        _semantic_measure_override(
            {**features, "semantic_neighbor_best_class": "dimension_value"},
            _SemanticOverrideSettings(
                mode="centroid-and-neighbor",
                min_centroid_score=0.80,
                min_margin=0.10,
                min_neighbor_vote=0.75,
            ),
        )
        is None
    )


def test_measure_context_features_compute_evidence_and_votes() -> None:
    config = load_config("configs/evaluation.yaml")
    rows = [
        {
            "term_id": "term_population",
            "canonical_term": "Population by sex",
            "matching_key": "population by sex",
            "occurrence_count": 2,
            "table_count": 1,
            "occurrence_ids_json": '["occ_title", "occ_value"]',
            "role_summary_json": '{"metadata_value": 1, "title_keyphrase": 1}',
            "metadata_columns_json": '["sex"]',
            "has_metadata_value_evidence": True,
            "has_percent": False,
            "has_unit_word": False,
            "has_age_pattern": False,
            "looks_like_dimension_name": False,
        }
    ]
    features = build_measure_context_features(
        config,
        rows,
        {
            "term_population": {
                "semantic_best_centroid_class": "measure",
                "semantic_neighbor_best_class": "measure",
                "semantic_centroid_measure": 0.82,
                "semantic_centroid_dimension_value": 0.70,
            }
        },
        term_occurrences=[
            {
                "term_id": "term_population",
                "occurrence_id": "occ_title",
                "source_area": "title_keyphrase",
                "metadata_column": None,
            },
            {
                "term_id": "term_population",
                "occurrence_id": "occ_value",
                "source_area": "metadata_value",
                "metadata_column": "sex",
            },
        ],
        title_terms=[
            {
                "occurrence_id": "occ_title",
                "extraction_rule": "title_keyphrase_statistical_head",
            }
        ],
    )

    context = features["term_population"]
    assert context["title_keyphrase_count"] == 1
    assert context["metadata_value_count"] == 1
    assert context["dimension_context_evidence"] is True
    assert context["measure_vote_count"] == 3
    assert context["negative_vote_count"] >= 1
    assert weak_supervision_vote_summary(features)["label_counts"]["measure"] == 1


def test_evidence_gated_semantic_measure_override_accepts_grounded_measure() -> None:
    semantic = {
        "semantic_best_centroid_class": "measure",
        "semantic_neighbor_best_class": "measure",
        "semantic_centroid_measure": 0.86,
        "semantic_centroid_other_ambiguous": 0.50,
        "semantic_centroid_dimension_value": 0.62,
        "semantic_neighbor_vote_measure": 0.80,
    }
    context = {
        "measure_acceptance_score": 0.80,
        "negative_dimension_score": 0.20,
        "measure_vote_agreement": 3,
        "title_keyphrase_count": 1,
        "title_clause_count": 0,
        "economic_measure_head": False,
        "count_quantity_measure_head": False,
        "statistical_head_match": True,
    }

    confidence, reason, subtype = _evidence_gated_semantic_measure_override(
        semantic,
        context,
        _CalibrationSettings(
            non_other_threshold=0.55,
            measure_threshold=0.60,
            override=_SemanticOverrideSettings(
                mode="centroid-and-neighbor",
                min_centroid_score=0.80,
                min_margin=0.10,
                min_neighbor_vote=0.75,
            ),
            evidence=_EvidenceGateSettings(
                measure_acceptance_min_score=0.70,
                max_negative_dimension_score=0.35,
                measure_vote_min_agreement=2,
            ),
        ),
        enabled=True,
    )

    assert confidence is not None
    assert reason is None
    assert subtype == "title_keyphrase_measure"


def test_evidence_gated_semantic_measure_override_rejects_negative_context() -> None:
    semantic = {
        "semantic_best_centroid_class": "measure",
        "semantic_neighbor_best_class": "measure",
        "semantic_centroid_measure": 0.86,
        "semantic_centroid_other_ambiguous": 0.50,
        "semantic_centroid_dimension_value": 0.62,
        "semantic_neighbor_vote_measure": 0.80,
    }
    context = {
        "measure_acceptance_score": 0.80,
        "negative_dimension_score": 0.60,
        "measure_vote_agreement": 3,
    }

    confidence, reason, subtype = _evidence_gated_semantic_measure_override(
        semantic,
        context,
        _CalibrationSettings(
            non_other_threshold=0.55,
            measure_threshold=0.60,
            override=_SemanticOverrideSettings(
                mode="centroid-and-neighbor",
                min_centroid_score=0.80,
                min_margin=0.10,
                min_neighbor_vote=0.75,
            ),
            evidence=_EvidenceGateSettings(
                measure_acceptance_min_score=0.70,
                max_negative_dimension_score=0.35,
                measure_vote_min_agreement=2,
            ),
        ),
        enabled=True,
    )

    assert confidence is None
    assert reason == "negative_dimension_score_above_threshold"
    assert subtype is None


def test_measure_subtype_assignment_is_diagnostic_only() -> None:
    subtype = measure_subtype_for_context(
        {
            "title_keyphrase_count": 0,
            "title_clause_count": 0,
            "economic_measure_head": True,
            "count_quantity_measure_head": False,
            "statistical_head_match": True,
        }
    )

    assert subtype == "economic_measure"


def test_settings_payload_contains_calibration_knobs() -> None:
    payload = _settings_payload(
        _CalibrationSettings(
            non_other_threshold=0.8,
            measure_threshold=0.9,
            override=_SemanticOverrideSettings(
                mode="centroid-and-neighbor",
                min_centroid_score=0.84,
                min_margin=0.1,
                min_neighbor_vote=1.0,
            ),
        )
    )

    assert payload["non_other_threshold"] == 0.8
    assert payload["measure_threshold"] == 0.9
    assert payload["semantic_override_mode"] == "centroid-and-neighbor"
    assert payload["measure_acceptance_min_score"] == 0.7


def test_generate_term_embeddings_reuses_matching_cache(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/evaluation.yaml")
    path = tmp_path / "term_embeddings.parquet"
    rows = [{"term_id": "term_a", "canonical_term": "population"}]
    write_parquet_rows(
        [
            {
                "term_id": "term_a",
                "canonical_term": "population",
                "model": config.classification.pearl_model,
                "revision": config.classification.pearl_revision,
                "embedding": [1.0, 0.0],
            }
        ],
        path,
    )

    def fail_loader():
        raise AssertionError("embedding model should not load when cache matches")

    monkeypatch.setattr("statvocab.classify_embeddings._sentence_transformer_class", fail_loader)

    assert generate_term_embeddings(config, rows, output_path=path) == path


def test_classification_evaluation_scores_current_exports(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import csv

    import statvocab.classification_evaluate as classification_evaluate

    config = load_config("configs/evaluation.yaml")
    config = config.model_copy(
        update={
            "paths": config.paths.model_copy(
                update={"outputs_dir": tmp_path / "outputs", "reports_dir": tmp_path / "report"}
            )
        }
    )
    config.paths.outputs_dir.mkdir(parents=True)
    fields = [
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
    for filename in (
        "measures.csv",
        "dimension_names.csv",
        "dimension_values.csv",
        "units.csv",
        "other_ambiguous.csv",
    ):
        with (config.paths.outputs_dir / filename).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            if filename == "measures.csv":
                writer.writerow(
                    {
                        "term_id": "term_a",
                        "canonical_term": "population",
                        "category": "measure",
                        "confidence": "1.0",
                        "variant": "test",
                        "protected": "true",
                        "evidence": "test",
                        "occurrence_ids_json": "[]",
                        "run_id": "run_test",
                    }
                )

    monkeypatch.setattr(
        classification_evaluate,
        "ensure_gold_templates",
        lambda config: (tmp_path / "sample.csv", tmp_path / "labels.csv", tmp_path / "relabel.csv"),
    )
    monkeypatch.setattr(
        classification_evaluate,
        "completed_gold_labels",
        lambda config: [
            {
                "term_id": "term_a",
                "canonical_term": "population",
                "category": "measure",
                "split": "final_test",
                "audit_source": "random_sample",
            }
        ],
    )

    _path, payload = evaluate_classification(config)

    assert payload["evaluated_prediction_source"] == "current_category_exports"
    assert payload["metrics"]["accuracy"] == 1.0
    assert payload["category_counts"]["measure"] == 1


def test_classification_gate_bootstraps_only_local_hybrid_on_stale_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import statvocab.classification as classification

    config = load_config("configs/evaluation.yaml")
    monkeypatch.setattr(classification, "completed_gold_labels", lambda config: [])
    existing = [{"term_id": "old_term", "category": VocabularyCategory.MEASURE.value}]
    candidate = [
        {"term_id": "new_term_a", "category": VocabularyCategory.MEASURE.value},
        {"term_id": "new_term_b", "category": VocabularyCategory.OTHER_AMBIGUOUS.value},
    ]

    local_gate = _gate_exports(
        config=config,
        variant="local-hybrid",
        existing_predictions=existing,
        candidate_predictions=candidate,
        metrics_payload={},
    )
    semantic_gate = _gate_exports(
        config=config,
        variant="semantic-hybrid",
        existing_predictions=existing,
        candidate_predictions=candidate,
        metrics_payload={},
    )

    assert local_gate["accepted"] is True
    assert local_gate["baseline_compatible"] is False
    assert semantic_gate["accepted"] is False
    assert "run local-hybrid" in semantic_gate["reasons"][-1]
