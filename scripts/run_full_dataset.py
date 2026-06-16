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
    TimeRemainingColumn,
)

from statvocab.config import load_config
from statvocab.full_corpus import (
    STAGE_SEQUENCE,
    ProgressEvent,
    audit_full_corpus_resume,
    run_full_corpus,
)


def _latest_incomplete_run(outputs_dir: Path) -> str | None:
    full_dir = outputs_dir / "full_corpus"
    if not full_dir.exists():
        return None
    candidates = [
        path
        for path in full_dir.iterdir()
        if path.is_dir()
        and (path / "stage_checkpoints.json").exists()
        and (
            not (path / "run_manifest.json").exists()
            or '"completed_with_blockers"'
            in (path / "run_manifest.json").read_text(encoding="utf-8")
        )
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
    parser.add_argument(
        "--show-copy-plan",
        action="store_true",
        help="Print the folders to copy when moving an in-progress full run to another computer.",
    )
    parser.add_argument(
        "--audit-resume",
        action="store_true",
        help="Inspect checkpoint reuse without executing any full-corpus stage.",
    )
    return parser


def _copy_plan(config_path: str, run_id: str, config: Any) -> dict[str, Any]:
    run_dir = config.paths.outputs_dir / "full_corpus" / run_id
    partial_dir = config.paths.processed_dir / "full_extract_partial" / run_id
    raw_dir = config.paths.raw_dir / (config.corpus.archive_name or "").removesuffix(".tgz")
    return {
        "config": config_path,
        "run_id": run_id,
        "copy_to_new_computer": [
            "git checkout the same branch/commit after pushing the code",
            str(raw_dir),
            str(run_dir),
            str(partial_dir),
        ],
        "dataset_note": "Copy the 100 GiB raw dataset manually; it is intentionally not in git.",
        "resume_command": (
            f".\\.venv\\Scripts\\python scripts\\run_full_dataset.py --config {config_path} "
            f"--resume-run-id {run_id}"
        ),
        "optional_if_extract_not_started": str(partial_dir),
    }


def main() -> int:
    args = _parser().parse_args()
    console = Console()
    config = load_config(args.config)
    run_id = args.resume_run_id
    resume = run_id is not None
    if run_id is None and not args.no_auto_resume:
        run_id = _latest_incomplete_run(config.paths.outputs_dir)
        resume = run_id is not None
    if args.show_copy_plan:
        if run_id is None:
            console.print(
                {
                    "error": (
                        "--show-copy-plan needs --resume-run-id or "
                        "an auto-detected incomplete run"
                    )
                }
            )
            return 2
        console.print(_copy_plan(args.config, run_id, config))
        return 0
    if args.audit_resume:
        if run_id is None:
            console.print({"error": "--audit-resume needs --resume-run-id or auto-detection"})
            return 2
        console.print(audit_full_corpus_resume(config, run_id=run_id))
        return 0

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
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("starting full-corpus run", total=stage_total)
        extract_task_id = progress.add_task(
            "extract tables",
            total=None,
            visible=False,
        )

        def on_progress(stage: str, event: ProgressEvent, checkpoint: dict[str, Any]) -> None:
            if event == "progress" and stage == "extract":
                total_tables = int(checkpoint.get("total_tables") or 0)
                completed_tables = int(checkpoint.get("completed_tables") or 0)
                failed_tables = int(checkpoint.get("failed_tables") or 0)
                resumed_tables = int(checkpoint.get("resumed_tables") or 0)
                fragment_bytes = int(checkpoint.get("fragment_bytes") or 0)
                tables_per_minute = checkpoint.get("tables_per_minute")
                eta_seconds = checkpoint.get("eta_seconds")
                table_id = str(checkpoint.get("table_id") or "")
                rate = (
                    f" rate={float(tables_per_minute):.2f}/min"
                    if tables_per_minute is not None
                    else ""
                )
                eta = (
                    f" eta={float(eta_seconds) / 60.0:.1f}m"
                    if eta_seconds is not None
                    else ""
                )
                progress.update(
                    extract_task_id,
                    total=total_tables or None,
                    completed=completed_tables,
                    visible=True,
                    description=(
                        "extract tables "
                        f"{completed_tables}/{total_tables} "
                        f"failed={failed_tables} resumed={resumed_tables} "
                        f"fragments={fragment_bytes}B{rate}{eta} {table_id}"
                    ),
                )
                progress.update(task_id, description="running extract")
                return
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
            if stage == "extract" and event == "finished":
                progress.update(extract_task_id, visible=False)

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
