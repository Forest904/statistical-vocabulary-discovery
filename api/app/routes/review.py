"""Human-in-the-loop review endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.app.routes.dependencies import get_state
from api.app.schemas import (
    ReviewAnswerRequest,
    ReviewAnswerResponse,
    ReviewCompileResponse,
    ReviewModeValue,
    ReviewStatsResponse,
    ReviewTaskResponse,
)
from api.app.state import ApiState
from statvocab.human_loop import (
    append_review_answer,
    compile_human_labels,
    next_review_task,
    review_stats,
)

router = APIRouter(prefix="/api/review", tags=["review"])
StateDependency = Annotated[ApiState, Depends(get_state)]


@router.get("/task", response_model=ReviewTaskResponse | None)
def review_task(
    state: StateDependency,
    mode: Annotated[ReviewModeValue, Query()] = "all",
) -> dict[str, Any] | None:
    """Return the next human-loop review task."""

    return next_review_task(state.config, mode=mode)


@router.post("/answer", response_model=ReviewAnswerResponse)
def review_answer(
    request: ReviewAnswerRequest,
    state: StateDependency,
) -> dict[str, Any]:
    """Append one human-loop review answer."""

    try:
        return append_review_answer(
            state.config,
            task_id=request.task_id,
            answer=request.answer,
            reviewer_id=request.reviewer_id or "local_user",
            notes=request.notes,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_review_answer",
                "message": str(exc),
                "details": {"task_id": request.task_id, "answer": request.answer},
            },
        ) from exc


@router.get("/stats", response_model=ReviewStatsResponse)
def review_progress(state: StateDependency) -> dict[str, Any]:
    """Return human-loop review progress."""

    return review_stats(state.config)


@router.post("/compile", response_model=ReviewCompileResponse)
def review_compile(state: StateDependency) -> dict[str, Any]:
    """Compile human-loop labels and refresh evaluation."""

    _path, payload = compile_human_labels(state.config)
    return payload
