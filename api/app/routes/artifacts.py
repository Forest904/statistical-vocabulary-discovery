"""Read-only artifact endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from api.app.routes.dependencies import get_state
from api.app.schemas import (
    ClusterListResponse,
    EvaluationResponse,
    RelationListResponse,
    TableDetailResponse,
    TermDetailResponse,
)
from api.app.state import ApiState

router = APIRouter(prefix="/api", tags=["artifacts"])
StateDependency = Annotated[ApiState, Depends(get_state)]
PageQuery = Annotated[int, Query(ge=1)]
PageSizeQuery = Annotated[int, Query(ge=1, le=100)]


@router.get("/tables/{table_id}", response_model=TableDetailResponse)
def table_detail(table_id: str, state: StateDependency) -> dict[str, object]:
    """Return one source-table detail payload."""

    payload = state.table_detail(table_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "table_not_found",
                "message": f"Unknown table ID: {table_id}",
                "details": {"table_id": table_id},
            },
        )
    return payload


@router.get("/terms/{term_id}", response_model=TermDetailResponse)
def term_detail(term_id: str, state: StateDependency) -> dict[str, object]:
    """Return one vocabulary-term detail payload."""

    payload = state.term_detail(term_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "term_not_found",
                "message": f"Unknown term ID: {term_id}",
                "details": {"term_id": term_id},
            },
        )
    return payload


@router.get("/clusters", response_model=ClusterListResponse)
def clusters(
    state: StateDependency,
    page: PageQuery = 1,
    page_size: PageSizeQuery = 20,
) -> dict[str, object]:
    """Return paginated measure-cluster summaries."""

    return state.cluster_list(page=page, page_size=page_size)


@router.get("/relations", response_model=RelationListResponse)
def relations(
    state: StateDependency,
    page: PageQuery = 1,
    page_size: PageSizeQuery = 20,
    term_id: str | None = None,
    relation_type: str | None = None,
) -> dict[str, object]:
    """Return paginated measure relations."""

    return state.relation_list(
        page=page,
        page_size=page_size,
        term_id=term_id,
        relation_type=relation_type,
    )


@router.get("/evaluation", response_model=EvaluationResponse)
def evaluation(state: StateDependency) -> dict[str, object]:
    """Return available evaluation summaries."""

    return {"summaries": state.evaluation}
