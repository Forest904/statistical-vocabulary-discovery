"""Runtime settings for the StatVocab API."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class ApiSettings(BaseModel):
    """Environment-backed API settings."""

    config_path: Path = Field(default=Path("configs/core.yaml"))


def load_settings() -> ApiSettings:
    """Load settings from environment variables."""

    return ApiSettings(
        config_path=Path(os.environ.get("STATVOCAB_CONFIG", "configs/core.yaml")),
    )
