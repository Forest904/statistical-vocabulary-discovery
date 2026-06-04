import pytest
from pydantic import ValidationError

from statvocab.classify_llm import validate_llm_response
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
