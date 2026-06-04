"""Stable identifiers and data contracts shared across pipeline stages."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import blake2b
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VocabularyCategory(StrEnum):
    """Final mutually exclusive vocabulary categories."""

    MEASURE = "measure"
    DIMENSION_NAME = "dimension_name"
    DIMENSION_VALUE = "dimension_value"
    UNIT = "unit"
    OTHER_AMBIGUOUS = "other_ambiguous"


class RelationType(StrEnum):
    """Allowed measure-to-measure relation labels."""

    BROADER_THAN = "broader_than"
    NARROWER_THAN = "narrower_than"
    VARIANT_OF = "variant_of"
    RELATED_TO = "related_to"


class EvidenceSource(StrEnum):
    """Known source surfaces for extracted evidence."""

    TITLE = "title"
    HEADER = "header"
    METADATA_COLUMN_NAME = "metadata_column_name"
    METADATA_VALUE = "metadata_value"
    RESOURCE = "resource"
    MODEL_OUTPUT = "model_output"


ID_PREFIXES: frozenset[str] = frozenset(
    {
        "cfg",
        "evidence",
        "occ",
        "relation",
        "resource",
        "run",
        "table",
        "term",
    }
)


def stable_id(prefix: str, *parts: object, digest_size: int = 10) -> str:
    """Create a deterministic lowercase identifier from semantic parts."""

    if prefix not in ID_PREFIXES:
        allowed = ", ".join(sorted(ID_PREFIXES))
        raise ValueError(f"Unknown stable ID prefix {prefix!r}; expected one of: {allowed}")
    normalized = "\x1f".join(str(part).strip().casefold() for part in parts)
    digest = blake2b(normalized.encode("utf-8"), digest_size=digest_size).hexdigest()
    return f"{prefix}_{digest}"


class ContractModel(BaseModel):
    """Base model with strict, JSON-serializable contracts."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class EvidenceRef(ContractModel):
    """Pointer to source evidence for a term, prediction, or relationship."""

    evidence_id: str = Field(..., min_length=1)
    table_id: str | None = None
    source: EvidenceSource
    raw_value: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)


class TermOccurrence(ContractModel):
    """One observed surface form in one source table."""

    occurrence_id: str
    table_id: str
    raw_value: str = Field(..., min_length=1)
    normalized_value: str = Field(..., min_length=1)
    source: EvidenceSource
    location: str = Field(..., min_length=1)


class VocabularyTerm(ContractModel):
    """Global term with category and occurrence provenance."""

    term_id: str
    canonical_term: str = Field(..., min_length=1)
    matching_key: str = Field(..., min_length=1)
    category: VocabularyCategory | None = None
    occurrence_ids: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("occurrence_ids")
    @classmethod
    def occurrence_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("occurrence_ids must be unique")
        return value


class MeasureRelation(ContractModel):
    """Grounded candidate semantic relation between two measure terms."""

    relation_id: str
    source_term_id: str
    target_term_id: str
    relation_type: RelationType
    evidence_ids: tuple[str, ...]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("target_term_id")
    @classmethod
    def reject_self_relation(cls, value: str, info: Any) -> str:
        source_term_id = info.data.get("source_term_id")
        if source_term_id == value:
            raise ValueError("measure relations cannot be self-relations")
        return value


class RunState(StrEnum):
    """Lifecycle state recorded in run manifests."""

    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RunManifest(ContractModel):
    """Immutable manifest header for one reproducible pipeline run."""

    run_id: str
    config_id: str
    state: RunState
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    corpus: Literal["fixture", "core", "full"]
    pipeline_stage: str = Field(..., min_length=1)
    resources: tuple[str, ...] = Field(default_factory=tuple)
    artifacts: tuple[str, ...] = Field(default_factory=tuple)

