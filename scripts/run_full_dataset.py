"""Run the full StatVocab dataset with progress and resume support."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from statvocab.config import load_config
from statvocab.full_corpus import STAGE_SEQUENCE, ProgressEvent, run_full_corpus


def _latest_incomplete_run(outputs_dir: Path) -> str | None:
    full_dir = outputs_dir / "full_corpus"
    if not full_dir.exists():
        return None
    candidates = [
        path
        for path in full_dir.iterdir()
        if path.is_dir()
        and (path / "stage_checkpoints.json").exists()
        and not (path / "run_manifest.json").exists()
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: (path / "stage_checkpoints.json").stat().st_mtime)
    return latest.name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full 7,605-table StatVocab dataset with stage progress.",
    )
    parser.add_argument(
        "--config",
        default="configs/full.yaml",
        help="Path to the full-corpus config.",
    )
    parser.add_argument(
        "--resume-run-id",
        default=None,
        help="Resume a specific outputs/full_corpus/<run_id> checkpoint directory.",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Start a new run even if an unfinished run directory exists.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    console = Console()
    config = load_config(args.config)
    run_id = args.resume_run_id
    resume = run_id is not None
    if run_id is None and not args.no_auto_resume:
        run_id = _latest_incomplete_run(config.paths.outputs_dir)
        resume = run_id is not None

    completed: set[str] = set()
    stage_total = len(STAGE_SEQUENCE)
    console.print(
        {
            "config": args.config,
            "resume": resume,
            "run_id": run_id or "new",
            "checkpoint_hint": str(config.paths.outputs_dir / "full_corpus"),
        }
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("starting full-corpus run", total=stage_total)

        def on_progress(stage: str, event: ProgressEvent, checkpoint: dict[str, Any]) -> None:
            status = str(checkpoint.get("status") or event)
            if event == "started":
                progress.update(task_id, description=f"running {stage}")
                return
            if event == "resumed":
                progress.update(task_id, description=f"resumed {stage}")
            else:
                progress.update(task_id, description=f"{stage}: {status}")
            if stage not in completed:
                completed.add(stage)
                progress.advance(task_id, 1)

        payload = run_full_corpus(
            config,
            run_id=run_id,
            resume=resume,
            progress_callback=on_progress,
        )

    console.print(payload)
    console.print(
        "Validate with: "
        f"statvocab validate-artifacts --config {args.config} --run-id {payload['run_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
