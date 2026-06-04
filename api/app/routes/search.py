"""Search endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.app.routes.dependencies import get_state
from api.app.schemas import SearchRequest, SearchResponse
from api.app.state import ApiState

router = APIRouter(prefix="/api", tags=["search"])
StateDependency = Annotated[ApiState, Depends(get_state)]


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, state: StateDependency) -> dict[str, object]:
    """Search source tables with grounded evidence."""

    return state.search(
        request.query,
        page=request.page,
        page_size=request.page_size,
        system=request.system,
    )
