"""Health endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.app.routes.dependencies import get_state
from api.app.schemas import HealthResponse
from api.app.state import ApiState

router = APIRouter(prefix="/api", tags=["health"])
StateDependency = Annotated[ApiState, Depends(get_state)]


@router.get("/health", response_model=HealthResponse)
def health(state: StateDependency) -> dict[str, object]:
    """Return loaded-run and artifact readiness."""

    return state.health()
