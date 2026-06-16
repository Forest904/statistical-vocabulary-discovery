"""FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.app.routes import artifacts, health, search
from api.app.settings import load_settings
from api.app.state import ApiStartupError, load_api_state


def _error_response(
    status_code: int,
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details or {}}},
    )


def create_app(*, load_on_startup: bool = True) -> FastAPI:
    """Create the FastAPI app."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if load_on_startup:
            settings = load_settings()
            app.state.api_state = load_api_state(settings)
        yield

    app = FastAPI(
        title="StatVocab API",
        version="0.1.0",
        description="Read-only APIs over grounded StatVocab artifacts and search indexes.",
        lifespan=lifespan,
    )
    app.include_router(search.router)
    app.include_router(artifacts.router)
    app.include_router(health.router)

    @app.exception_handler(ApiStartupError)
    async def startup_error_handler(_request: Request, exc: ApiStartupError) -> JSONResponse:
        return _error_response(
            503,
            code=exc.code,
            message=str(exc),
            details=exc.details,
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = cast(Any, exc.detail)
        if isinstance(detail, dict) and "code" in detail:
            return _error_response(
                exc.status_code,
                code=str(detail["code"]),
                message=str(detail.get("message", detail["code"])),
                details=dict(detail.get("details") or {}),
            )
        return _error_response(
            exc.status_code,
            code="http_error",
            message=str(detail),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            422,
            code="validation_error",
            message="Request validation failed.",
            details={"errors": exc.errors()},
        )

    return app


app = create_app()
