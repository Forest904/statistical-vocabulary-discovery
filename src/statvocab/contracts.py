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


class ResourceValidationStatus(StrEnum):
    """Validation state for acquired or reused source resources."""

    VERIFIED = "verified"
    REUSED = "reused"
    DOWNLOADED = "downloaded"
    MISSING = "missing"
    FAILED = "failed"
    NOT_VERIFIED_ARCHIVE_ABSENT = "not_verified_archive_absent"


class ParseStatus(StrEnum):
    """Table-level ingestion outcome."""

    PARSED = "parsed"
    WARNING = "warning"
    FAILED = "failed"


ID_PREFIXES: frozenset[str] = frozenset(
    {
        "cfg",
        "evidence",
        "geo",
        "occ",
        "relation",
        "resource",
        "run",
        "string",
        "table",
        "term",
        "time",
    }
)


class TimeGranularity(StrEnum):
    """Supported normalized time granularities."""

    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    DAY = "day"
    RANGE = "range"
    OTHER_PERIOD = "other_period"


class ExtractionSourceArea(StrEnum):
    """Source surface used by Milestone 2 extraction artifacts."""

    HEADER_TIME = "header_time"
    TITLE = "title"
    HEADER_NAME = "header_name"
    METADATA_VALUE = "metadata_value"
    TITLE_FULL = "title_full"
    TITLE_CLAUSE = "title_clause"


class GeographyVariant(StrEnum):
    """Geography dictionary variants produced by Milestone 2."""

    NUTS = "nuts"
    ENHANCED = "enhanced"


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


class ResourceRecord(ContractModel):
    """Provenance and validation metadata for one local resource."""

    resource_id: str
    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    local_path: str = Field(..., min_length=1)
    license: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    expected_md5: str | None = None
    expected_size_bytes: int | None = Field(default=None, ge=0)
    checksum_md5: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    validation_status: ResourceValidationStatus
    message: str = ""


class TableReference(ContractModel):
    """One source table file selected for ingestion."""

    table_id: str
    filename: str
    local_path: str
    original_local_path: str | None = None
    title: str | None = None
    evidence_url: str
    file_md5: str
    original_file_md5: str | None = None
    size_bytes: int = Field(ge=0)
    original_size_bytes: int | None = Field(default=None, ge=0)
    source_repaired: bool = False
    repair_reason: str | None = None


class SourceRepair(ContractModel):
    """Documented source-file repair applied during ingestion."""

    table_id: str
    original_path: str
    original_md5: str
    original_size_bytes: int = Field(ge=0)
    defect_reason: str = Field(..., min_length=1)
    replacement_path: str
    replacement_md5: str
    replacement_size_bytes: int = Field(ge=0)
    replacement_source_corpus: str = Field(..., min_length=1)
    repair_status: Literal["applied", "missing_replacement", "invalid_replacement"]
    repaired_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    message: str = ""


class ParsedColumn(ContractModel):
    """One parsed source column."""

    name: str
    index: int = Field(ge=0)
    role: Literal["metadata", "time"]


class ParsedObservation(ContractModel):
    """One parsed observation cell."""

    row_number: int = Field(ge=1)
    column_name: str
    raw_value: str
    numeric_value: float | None = None
    missing: bool = False
    flag: str | None = None


class ParseWarning(ContractModel):
    """Non-fatal or fatal parse issue attached to a table."""

    table_id: str
    row_number: int | None = Field(default=None, ge=1)
    reason: str = Field(..., min_length=1)
    expected_columns: int | None = Field(default=None, ge=0)
    actual_columns: int | None = Field(default=None, ge=0)


class ParsedTable(ContractModel):
    """Ingestion summary for one table."""

    reference: TableReference
    columns: tuple[ParsedColumn, ...]
    row_count: int = Field(ge=0)
    observation_count: int = Field(ge=0)
    malformed_row_count: int = Field(ge=0)
    warnings: tuple[ParseWarning, ...] = Field(default_factory=tuple)
    parse_status: ParseStatus
