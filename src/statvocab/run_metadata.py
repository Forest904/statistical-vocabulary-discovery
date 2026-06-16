"""Run identity and environment metadata helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from statvocab import __version__
from statvocab.config import AppConfig


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_metadata() -> dict[str, Any]:
    """Return Git metadata without blocking dirty worktree usage."""

    commit = _git_output(["rev-parse", "HEAD"])
    status = _git_output(["status", "--porcelain"])
    dirty = bool(status)
    return {
        "commit": commit or "",
        "dirty": dirty,
        "dirty_warning": "worktree has uncommitted changes" if dirty else "",
    }


def run_environment(config: AppConfig) -> dict[str, Any]:
    """Return reproducibility metadata attached to manifests and checkpoints."""

    git = git_metadata()
    return {
        "config_fingerprint": config.config_fingerprint,
        "config_summary": config.config_summary,
        "python_version": sys.version.split()[0],
        "package_version": __version__,
        "git": git,
        "corpus": {
            "name": config.corpus.name,
            "archive_name": config.corpus.archive_name or "",
            "expected_table_count": config.corpus.expected_table_count,
            "expected_md5": config.corpus.expected_md5 or "",
        },
    }
