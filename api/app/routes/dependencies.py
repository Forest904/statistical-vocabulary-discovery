"""FastAPI dependencies."""

from __future__ import annotations

from typing import cast

from fastapi import Request

from api.app.state import ApiState


def get_state(request: Request) -> ApiState:
    """Return loaded API state."""

    return cast(ApiState, request.app.state.api_state)
