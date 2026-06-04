"""Run manifest helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from statvocab.config import AppConfig
from statvocab.contracts import RunManifest, RunState, stable_id


def create_manifest(config: AppConfig, pipeline_stage: str) -> RunManifest:
    """Create a started manifest for the requested config and pipeline stage."""

    config_id = stable_id("cfg", *config.config_id_parts)
    run_id = stable_id("run", config_id, pipeline_stage, datetime.now(UTC).isoformat())
    return RunManifest(
        run_id=run_id,
        config_id=config_id,
        state=RunState.STARTED,
        corpus=config.corpus.name,
        pipeline_stage=pipeline_stage,
    )


def complete_manifest(
    manifest: RunManifest,
    *,
    resources: tuple[str, ...] = (),
    artifacts: tuple[str, ...] = (),
) -> RunManifest:
    """Return a completed copy of a manifest."""

    return manifest.model_copy(
        update={
            "state": RunState.SUCCEEDED,
            "finished_at": datetime.now(UTC),
            "resources": resources,
            "artifacts": artifacts,
        }
    )


def write_manifest(manifest: RunManifest, output_path: str | Path) -> Path:
    """Write a manifest as stable, indented JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

