"""Public API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    """Stable API error payload."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Stable API error response envelope."""

    error: ErrorBody


class Pagination(BaseModel):
    """Page-number pagination metadata."""

    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total: int = Field(ge=0)


SearchSystem = Literal["fused", "title_bm25", "all_vocabulary_bm25", "pearl_semantic"]
RelationTypeValue = Literal["broader_than", "narrower_than", "variant_of", "related_to"]


class SearchRequest(BaseModel):
    """Search request body."""

    query: str = Field(min_length=1)
    page: int = Field(default=1, ge=1, le=1000)
    page_size: int = Field(default=20, ge=1, le=100)
    system: SearchSystem = "fused"


class SearchResponse(BaseModel):
    """Grounded table-search response."""

    query: dict[str, Any]
    notice: str
    pagination: Pagination
    results: list[dict[str, Any]]


class HealthResponse(BaseModel):
    """API readiness response."""

    status: Literal["ok", "degraded"]
    config_name: str
    corpus: str
    loaded_search_run_id: str
    document_count: int
    readiness: dict[str, bool]
    artifact_counts: dict[str, int]
    warnings: list[str]


class TableDetailResponse(BaseModel):
    """Denormalized source-table detail."""

    table_id: str
    title: str
    source_url: str
    parse_status: str
    metadata: dict[str, Any]
    vocabulary: dict[str, Any]
    geographies: list[dict[str, Any]]
    times: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    related_terms: list[dict[str, Any]]


class TermDetailResponse(BaseModel):
    """Denormalized vocabulary-term detail."""

    term_id: str
    canonical_term: str
    category: str | None = None
    confidence: float | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    occurrence_ids: list[str] = Field(default_factory=list)
    table_appearances: list[dict[str, Any]] = Field(default_factory=list)
    cluster: dict[str, Any] | None = None
    relations: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class ClusterListResponse(BaseModel):
    """Paginated cluster summaries."""

    pagination: Pagination
    items: list[dict[str, Any]]


class RelationListResponse(BaseModel):
    """Paginated relation summaries."""

    pagination: Pagination
    items: list[dict[str, Any]]


class EvaluationResponse(BaseModel):
    """Read-only evaluation summary response."""

    summaries: dict[str, dict[str, Any]]
