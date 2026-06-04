"""Strict validation scaffolding for optional paid classification adjudication."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from statvocab.contracts import VocabularyCategory


class LlmClassificationResponse(BaseModel):
    """Allowed JSON shape for one model-suggested term classification."""

    model_config = ConfigDict(extra="forbid")

    term_id: str = Field(..., min_length=1)
    category: VocabularyCategory
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(..., min_length=1, max_length=600)

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_ids must be unique")
        return value


def validate_llm_response(
    payload: dict[str, Any],
    *,
    allowed_term_ids: set[str],
    allowed_evidence_ids_by_term: dict[str, set[str]],
) -> LlmClassificationResponse:
    """Validate schema, vocabulary membership, and evidence membership."""

    try:
        response = LlmClassificationResponse.model_validate(payload)
    except ValidationError:
        raise

    if response.term_id not in allowed_term_ids:
        raise ValueError(f"Unknown term_id from model output: {response.term_id}")
    allowed_evidence_ids = allowed_evidence_ids_by_term.get(response.term_id, set())
    unknown_evidence = sorted(set(response.evidence_ids) - allowed_evidence_ids)
    if unknown_evidence:
        joined = ", ".join(unknown_evidence)
        raise ValueError(f"Model output referenced evidence outside the term: {joined}")
    return response


def response_schema() -> dict[str, Any]:
    """Return a JSON-schema dictionary suitable for prompt documentation."""

    return LlmClassificationResponse.model_json_schema()
