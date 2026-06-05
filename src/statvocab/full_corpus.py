"""Full-corpus run orchestration and scale measurements."""

from __future__ import annotations

import csv
import json
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from statvocab.classification import run_classification
from statvocab.cluster_measures import run_measure_clustering
from statvocab.config import AppConfig
from statvocab.contracts import ResourceRecord, ResourceValidationStatus
from statvocab.ingest import run_ingestion
from statvocab.manifests import complete_manifest, create_manifest, write_manifest
from statvocab.relations import run_measure_relations
from statvocab.resources import (
    acquire_resources,
    count_csv_files,
    file_md5,
    write_resource_manifest,
)
from statvocab.retrieval_evaluate import evaluate_retrieval
from statvocab.search.documents import build_search_documents
from statvocab.search.engine import build_search_index
from statvocab.search.lexical import build_lexical_index
from statvocab.vocabulary import run_extraction

StageStatus = Literal["succeeded", "failed", "skipped", "blocked"]
ProgressEvent = Literal["started", "finished", "resumed"]
ProgressCallback = Callable[[str, ProgressEvent, dict[str, Any]], None]
CHECKSUM_SIZE_LIMIT_BYTES = 512 * 1024 * 1024
MIN_PROJECTED_ARTIFACT_BYTES = 10 * 1024 * 1024 * 1024
STAGE_SEQUENCE = (
    "disk-preflight",
    "acquire",
    "ingest",
    "extract",
    "classify-local-hybrid",
    "cluster-measures",
    "relations",
    "build-search-index",
    "evaluate-retrieval",
)


class PeakMemorySampler:
    """Sample process RSS while a stage runs."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: Any | None = None

    def __enter__(self) -> PeakMemorySampler:
        try:
            psutil = import_module("psutil")
            self._process = psutil.Process()
        except ModuleNotFoundError:
            return self

        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._record()

    def _record(self) -> None:
        if self._process is None:
            return
        rss = int(self._process.memory_info().rss)
        self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._record()
            time.sleep(self.interval_seconds)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def disk_snapshot(path: Path) -> dict[str, int | str]:
    """Return disk usage for the drive containing ``path``."""

    target = _existing_parent(path)
    usage = shutil.disk_usage(target)
    return {
        "path": str(target),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def directory_size(path: Path) -> int:
    """Return total file bytes under a directory."""

    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def artifact_info(path: Path) -> dict[str, Any]:
    """Collect existence, size, and optional checksum metadata for an artifact."""

    if not path.exists():
        return {"path": str(path), "exists": False, "size_bytes": 0, "md5": None}
    if path.is_dir():
        return {
            "path": str(path),
            "exists": True,
            "kind": "directory",
            "size_bytes": directory_size(path),
            "md5": None,
        }
    size = path.stat().st_size
    checksum = file_md5(path) if size <= CHECKSUM_SIZE_LIMIT_BYTES else None
    return {
        "path": str(path),
        "exists": True,
        "kind": "file",
        "size_bytes": size,
        "md5": checksum,
    }


def artifact_infos(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Collect artifact metadata, preserving path order and removing duplicates."""

    seen: set[str] = set()
    infos: list[dict[str, Any]] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        infos.append(artifact_info(path))
    return infos


def _csv_count_and_size(path: Path) -> tuple[int, int]:
    if not path.exists() or not path.is_dir():
        return 0, 0
    count = 0
    total = 0
    for child in path.glob("*.csv"):
        if child.is_file():
            count += 1
            total += child.stat().st_size
    return count, total


def disk_preflight(config: AppConfig) -> dict[str, Any]:
    """Check disk headroom and full-corpus resource state before heavy work."""

    archive_name = config.corpus.archive_name or ""
    archive_path = config.paths.raw_dir / archive_name
    extracted_dir = config.paths.raw_dir / archive_name.removesuffix(".tgz")
    extracted_count, extracted_size = _csv_count_and_size(extracted_dir)
    archive_size = archive_path.stat().st_size if archive_path.exists() else 0
    free_bytes = int(disk_snapshot(config.paths.raw_dir)["free_bytes"])
    projected_artifacts = max(MIN_PROJECTED_ARTIFACT_BYTES, extracted_size // 5)
    required_free = extracted_size + projected_artifacts
    if extracted_count == 0 and archive_size:
        required_free = max(archive_size * 4, MIN_PROJECTED_ARTIFACT_BYTES)
    if free_bytes < required_free:
        raise RuntimeError(
            "insufficient disk space for full-corpus run: "
            f"free={free_bytes}, required={required_free}"
        )

    warnings: list[str] = []
    if config.corpus.name == "full" and not archive_path.exists():
        if extracted_count == config.corpus.expected_table_count:
            warnings.append(
                "full archive is absent; continuing from exact-size extracted cache without "
                "archive checksum verification"
            )
        else:
            raise RuntimeError(
                f"missing {archive_path} and extracted cache has {extracted_count} CSV files; "
                f"expected {config.corpus.expected_table_count}"
            )

    return {
        "archive_path": str(archive_path),
        "archive_exists": archive_path.exists(),
        "archive_size_bytes": archive_size,
        "extracted_dir": str(extracted_dir),
        "extracted_csv_count": extracted_count,
        "expected_table_count": config.corpus.expected_table_count,
        "extracted_size_bytes": extracted_size,
        "free_bytes": free_bytes,
        "required_free_bytes": required_free,
        "warnings": warnings,
    }


def _stage_artifact_paths(result: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    artifacts = result.get("artifacts")
    if isinstance(artifacts, dict):
        paths.extend(Path(str(path)) for path in artifacts.values())
    elif isinstance(artifacts, list | tuple):
        paths.extend(Path(str(path)) for path in artifacts)
    manifest_path = result.get("manifest_path")
    if manifest_path:
        paths.append(Path(str(manifest_path)))
    metrics_path = result.get("metrics_path")
    if metrics_path:
        paths.append(Path(str(metrics_path)))
    return paths


def _coerce_stage_result(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, tuple) and len(raw) == 4:
        artifacts, _diagnostics_path, manifest_path, diagnostics = raw
        return {"artifacts": artifacts, "manifest_path": manifest_path, "diagnostics": diagnostics}
    if isinstance(raw, tuple) and len(raw) == 3:
        first, second, third = raw
        if isinstance(first, Path):
            return {"metrics_path": first, "diagnostics": second}
        return {"artifacts": first, "manifest_path": second, "diagnostics": third}
    raise TypeError(f"Unsupported stage result: {type(raw).__name__}")


def _last_valid_checkpoint(checkpoints: list[dict[str, Any]]) -> str | None:
    for checkpoint in reversed(checkpoints):
        if checkpoint.get("status") == "succeeded":
            return str(checkpoint["stage"])
    return None


def _load_checkpoints(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))


def _checkpoint_for_stage(
    checkpoints: list[dict[str, Any]],
    stage: str,
) -> dict[str, Any] | None:
    for checkpoint in reversed(checkpoints):
        if checkpoint.get("stage") == stage:
            return checkpoint
    return None


def _completed_checkpoint(
    checkpoints: list[dict[str, Any]],
    stage: str,
) -> dict[str, Any] | None:
    checkpoint = _checkpoint_for_stage(checkpoints, stage)
    if checkpoint and checkpoint.get("status") == "succeeded":
        return checkpoint
    return None


def _run_stage(
    *,
    name: str,
    config: AppConfig,
    checkpoints: list[dict[str, Any]],
    action: Callable[[], object],
    optional: bool = False,
    checkpoint_path: Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if progress_callback is not None:
        progress_callback(name, "started", {})
    started_at = _now_iso()
    disk_before = disk_snapshot(config.paths.data_dir)
    started = time.perf_counter()
    raw_result: object | None = None
    status: StageStatus = "succeeded"
    failure_message = ""
    try:
        with PeakMemorySampler() as memory:
            raw_result = action()
        result = _coerce_stage_result(raw_result)
        status = cast(StageStatus, result.get("status", "succeeded"))
        failure_message = str(result.get("failure_message") or "")
        peak_rss = memory.peak_rss_bytes
    except Exception as exc:
        result = {}
        status = "blocked" if optional else "failed"
        failure_message = str(exc)
        peak_rss = None

    finished_at = _now_iso()
    artifact_paths = _stage_artifact_paths(result)
    checkpoint = {
        "stage": name,
        "status": status,
        "optional": optional,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_clock_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss,
        "disk_before": disk_before,
        "disk_after": disk_snapshot(config.paths.data_dir),
        "run_id": result.get("diagnostics", {}).get("run_id") if isinstance(result, dict) else None,
        "manifest_path": str(result.get("manifest_path")) if result.get("manifest_path") else "",
        "artifacts": artifact_infos(artifact_paths),
        "failure_message": failure_message,
        "last_valid_checkpoint": _last_valid_checkpoint(checkpoints) or "none",
        "diagnostics": result.get("diagnostics", {}),
    }
    checkpoints.append(checkpoint)
    if checkpoint_path is not None:
        _write_checkpoints(checkpoints, checkpoint_path)
    if progress_callback is not None:
        progress_callback(name, "finished", checkpoint)
    return checkpoint, result if status == "succeeded" else None


def skipped_checkpoint(
    *,
    name: str,
    config: AppConfig,
    checkpoints: list[dict[str, Any]],
    reason: str,
    optional: bool = False,
    checkpoint_path: Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Append a skipped-stage checkpoint."""

    if progress_callback is not None:
        progress_callback(name, "started", {})
    snapshot = disk_snapshot(config.paths.data_dir)
    checkpoint = {
        "stage": name,
        "status": "skipped",
        "optional": optional,
        "started_at": _now_iso(),
        "finished_at": _now_iso(),
        "wall_clock_seconds": 0.0,
        "peak_rss_bytes": None,
        "disk_before": snapshot,
        "disk_after": snapshot,
        "run_id": None,
        "manifest_path": "",
        "artifacts": [],
        "failure_message": reason,
        "last_valid_checkpoint": _last_valid_checkpoint(checkpoints) or "none",
        "diagnostics": {},
    }
    checkpoints.append(checkpoint)
    if checkpoint_path is not None:
        _write_checkpoints(checkpoints, checkpoint_path)
    if progress_callback is not None:
        progress_callback(name, "finished", checkpoint)
    return checkpoint


def resumed_checkpoint(
    checkpoints: list[dict[str, Any]],
    stage: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any] | None:
    """Return a completed checkpoint and emit a progress event for resumed stages."""

    checkpoint = _completed_checkpoint(checkpoints, stage)
    if checkpoint is not None and progress_callback is not None:
        progress_callback(stage, "resumed", checkpoint)
    return checkpoint


def _acquire_stage(config: AppConfig) -> dict[str, Any]:
    manifest = create_manifest(config, "acquire")
    records = acquire_resources(config)
    resource_manifest_path = write_resource_manifest(
        records,
        config.paths.processed_dir / "resource_manifest.json",
    )
    completed = complete_manifest(
        manifest,
        resources=tuple(record.resource_id for record in records),
        artifacts=(str(resource_manifest_path),),
    )
    manifest_path = write_manifest(
        completed,
        config.paths.outputs_dir / "manifests" / f"{manifest.run_id}_acquire.json",
    )
    archive_records = [record for record in records if record.name.startswith("zenodo_")]
    warnings = [
        record.message
        for record in archive_records
        if record.validation_status == ResourceValidationStatus.NOT_VERIFIED_ARCHIVE_ABSENT
    ]
    diagnostics = {
        "run_id": manifest.run_id,
        "resource_count": len(records),
        "resources": {record.name: record.validation_status.value for record in records},
        "warnings": warnings,
        "extracted_table_count": count_csv_files(
            config.paths.raw_dir / (config.corpus.archive_name or "").removesuffix(".tgz")
        ),
    }
    return {
        "artifacts": {"resource_manifest": resource_manifest_path},
        "manifest_path": manifest_path,
        "diagnostics": diagnostics,
        "resources": records,
    }


def _load_resource_records(path: Path) -> tuple[ResourceRecord, ...]:
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(ResourceRecord.model_validate(item) for item in payload)


def _search_stage(config: AppConfig, resources: tuple[ResourceRecord, ...]) -> dict[str, Any]:
    try:
        artifacts, manifest_path, diagnostics = build_search_index(config, resources=resources)
        return {"artifacts": artifacts, "manifest_path": manifest_path, "diagnostics": diagnostics}
    except RuntimeError as exc:
        manifest = create_manifest(config, "build-search-index-lexical-only")
        output_dir = config.paths.outputs_dir / "search" / manifest.run_id
        rows, documents_path = build_search_documents(config, run_id=manifest.run_id)
        lexical_path = build_lexical_index(rows, output_dir / "lexical.sqlite")
        summary_path = output_dir / "search_index_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "run_id": manifest.run_id,
            "corpus": config.corpus.name,
            "document_count": len(rows),
            "semantic_status": "blocked",
            "semantic_blocker": str(exc),
            "lexical_index_size_bytes": lexical_path.stat().st_size if lexical_path.exists() else 0,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts = {
            "search_documents": documents_path,
            "lexical_index": lexical_path,
            "summary": summary_path,
        }
        completed = complete_manifest(
            manifest,
            resources=tuple(record.resource_id for record in resources),
            artifacts=tuple(str(path) for path in artifacts.values()),
        )
        manifest_path = write_manifest(
            completed,
            config.paths.outputs_dir / "manifests" / f"{manifest.run_id}_build_search_lexical.json",
        )
        return {
            "status": "blocked",
            "failure_message": f"semantic search blocked; lexical artifacts preserved: {exc}",
            "artifacts": artifacts,
            "manifest_path": manifest_path,
            "diagnostics": summary,
        }


def _retrieval_stage(config: AppConfig) -> dict[str, Any]:
    metrics_path, diagnostics = evaluate_retrieval(config)
    return {"metrics_path": metrics_path, "diagnostics": diagnostics}


def _write_checkpoints(checkpoints: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoints, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_resource_measurements(checkpoints: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "status",
        "wall_clock_seconds",
        "peak_rss_bytes",
        "disk_free_before_bytes",
        "disk_free_after_bytes",
        "artifact_size_bytes",
        "failure_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for checkpoint in checkpoints:
            artifact_size = sum(
                int(item.get("size_bytes") or 0) for item in checkpoint["artifacts"]
            )
            writer.writerow(
                {
                    "stage": checkpoint["stage"],
                    "status": checkpoint["status"],
                    "wall_clock_seconds": f"{float(checkpoint['wall_clock_seconds']):.6f}",
                    "peak_rss_bytes": checkpoint["peak_rss_bytes"] or "",
                    "disk_free_before_bytes": checkpoint["disk_before"]["free_bytes"],
                    "disk_free_after_bytes": checkpoint["disk_after"]["free_bytes"],
                    "artifact_size_bytes": artifact_size,
                    "failure_message": checkpoint["failure_message"],
                }
            )
    return path


def _write_scalability_report(
    *,
    run_id: str,
    checkpoints: list[dict[str, Any]],
    path: Path,
) -> Path:
    lines = [
        "# Full-Corpus Scalability And Bottleneck Report",
        "",
        f"Full-corpus run ID: `{run_id}`",
        "",
        "## Stage Summary",
        "",
        "| Stage | Status | Seconds | Peak RSS bytes | Artifact bytes | Blocker |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for checkpoint in checkpoints:
        artifact_size = sum(int(item.get("size_bytes") or 0) for item in checkpoint["artifacts"])
        blocker = str(checkpoint.get("failure_message") or "").replace("\n", " ")
        peak_rss = checkpoint["peak_rss_bytes"] if checkpoint["peak_rss_bytes"] is not None else ""
        lines.append(
            "| "
            f"{checkpoint['stage']} | {checkpoint['status']} | "
            f"{float(checkpoint['wall_clock_seconds']):.2f} | {peak_rss} | "
            f"{artifact_size} | {blocker} |"
        )
    lines.extend(
        [
            "",
            "## Bottleneck Policy",
            "",
            "No algorithmic optimization is applied by this orchestration layer. "
            "Stages marked failed, blocked, or unusually expensive in the table above are the "
            "evidence source for any later optimization.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_full_manifest(
    *,
    config: AppConfig,
    run_id: str,
    manifest_path: Path,
    checkpoints: list[dict[str, Any]],
    deliverables: dict[str, Path],
) -> Path:
    status_counts: dict[str, int] = {}
    for checkpoint in checkpoints:
        status = str(checkpoint["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    payload = {
        "run_id": run_id,
        "config_name": config.config_name,
        "corpus": config.corpus.name,
        "pipeline_stage": "run-all",
        "started_at": checkpoints[0]["started_at"] if checkpoints else _now_iso(),
        "finished_at": _now_iso(),
        "overall_status": (
            "completed_with_blockers"
            if any(
                checkpoint["status"] in {"failed", "blocked", "skipped"}
                for checkpoint in checkpoints
            )
            else "completed"
        ),
        "stage_status_counts": status_counts,
        "deliverables": {name: str(path) for name, path in deliverables.items()},
        "stage_checkpoints": str(deliverables["stage_checkpoints"]),
        "resource_measurements": str(deliverables["resource_measurements"]),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def run_full_corpus(
    config: AppConfig,
    *,
    classification_variant: str = "local-hybrid",
    run_id: str | None = None,
    resume: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the full-corpus scale attempt and write checkpoint/report deliverables."""

    manifest = create_manifest(config, "run-all")
    actual_run_id = run_id or manifest.run_id
    if run_id is not None:
        manifest = manifest.model_copy(update={"run_id": run_id})
    output_dir = config.paths.outputs_dir / "full_corpus" / actual_run_id
    checkpoints_path = output_dir / "stage_checkpoints.json"
    measurements_path = output_dir / "resource_measurements.csv"
    checkpoints: list[dict[str, Any]] = _load_checkpoints(checkpoints_path) if resume else []
    resources: tuple[ResourceRecord, ...] = ()

    preflight_checkpoint = resumed_checkpoint(
        checkpoints,
        "disk-preflight",
        progress_callback,
    )
    if preflight_checkpoint is None:
        preflight_checkpoint, _preflight_result = _run_stage(
            name="disk-preflight",
            config=config,
            checkpoints=checkpoints,
            action=lambda: {"diagnostics": disk_preflight(config)},
            checkpoint_path=checkpoints_path,
            progress_callback=progress_callback,
        )

    if preflight_checkpoint["status"] == "succeeded":
        acquire_checkpoint = resumed_checkpoint(checkpoints, "acquire", progress_callback)
        if acquire_checkpoint is not None:
            acquire_result = None
        else:
            acquire_checkpoint, acquire_result = _run_stage(
                name="acquire",
                config=config,
                checkpoints=checkpoints,
                action=lambda: _acquire_stage(config),
                checkpoint_path=checkpoints_path,
                progress_callback=progress_callback,
            )
    else:
        acquire_checkpoint = resumed_checkpoint(checkpoints, "acquire", progress_callback)
        if acquire_checkpoint is None:
            acquire_checkpoint = skipped_checkpoint(
                name="acquire",
                config=config,
                checkpoints=checkpoints,
                reason="disk preflight did not pass",
                checkpoint_path=checkpoints_path,
                progress_callback=progress_callback,
            )
        acquire_result = None
    if acquire_result and "resources" in acquire_result:
        resources = tuple(cast(tuple[ResourceRecord, ...], acquire_result["resources"]))
    else:
        resources = _load_resource_records(config.paths.processed_dir / "resource_manifest.json")

    can_ingest = checkpoints[0]["status"] == "succeeded" and (
        acquire_checkpoint["status"] == "succeeded"
        or count_csv_files(
            config.paths.raw_dir / (config.corpus.archive_name or "").removesuffix(".tgz")
        )
        == config.corpus.expected_table_count
    )
    if can_ingest:
        ingest_checkpoint = resumed_checkpoint(checkpoints, "ingest", progress_callback)
        if ingest_checkpoint is None:
            ingest_checkpoint, _ingest_result = _run_stage(
                name="ingest",
                config=config,
                checkpoints=checkpoints,
                action=lambda: run_ingestion(config, resources=resources),
                checkpoint_path=checkpoints_path,
                progress_callback=progress_callback,
            )
    else:
        ingest_checkpoint = resumed_checkpoint(checkpoints, "ingest", progress_callback)
        if ingest_checkpoint is None:
            ingest_checkpoint = skipped_checkpoint(
                name="ingest",
                config=config,
                checkpoints=checkpoints,
                reason="raw corpus is not available or disk preflight failed",
                checkpoint_path=checkpoints_path,
                progress_callback=progress_callback,
            )

    if ingest_checkpoint["status"] == "succeeded":
        extract_checkpoint = resumed_checkpoint(checkpoints, "extract", progress_callback)
        if extract_checkpoint is None:
            extract_checkpoint, _extract_result = _run_stage(
                name="extract",
                config=config,
                checkpoints=checkpoints,
                action=lambda: run_extraction(config, resources=resources),
                checkpoint_path=checkpoints_path,
                progress_callback=progress_callback,
            )
    else:
        extract_checkpoint = resumed_checkpoint(checkpoints, "extract", progress_callback)
        if extract_checkpoint is None:
            extract_checkpoint = skipped_checkpoint(
                name="extract",
                config=config,
                checkpoints=checkpoints,
                reason="ingestion did not complete in this run",
                checkpoint_path=checkpoints_path,
                progress_callback=progress_callback,
            )

    classify_stage = f"classify-{classification_variant}"
    if extract_checkpoint["status"] == "succeeded":
        classify_checkpoint = resumed_checkpoint(checkpoints, classify_stage, progress_callback)
        if classify_checkpoint is None:
            classify_checkpoint, _classify_result = _run_stage(
                name=classify_stage,
                config=config,
                checkpoints=checkpoints,
                action=lambda: run_classification(
                    config,
                    variant=classification_variant,
                    resources=resources,
                ),
                checkpoint_path=checkpoints_path,
                progress_callback=progress_callback,
            )
    else:
        classify_checkpoint = resumed_checkpoint(checkpoints, classify_stage, progress_callback)
        if classify_checkpoint is None:
            classify_checkpoint = skipped_checkpoint(
                name=classify_stage,
                config=config,
                checkpoints=checkpoints,
                reason="extraction did not complete in this run",
                checkpoint_path=checkpoints_path,
                progress_callback=progress_callback,
            )

    if classify_checkpoint["status"] == "succeeded":
        cluster_checkpoint = resumed_checkpoint(checkpoints, "cluster-measures", progress_callback)
        if cluster_checkpoint is None:
            cluster_checkpoint, _cluster_result = _run_stage(
                name="cluster-measures",
                config=config,
                checkpoints=checkpoints,
                action=lambda: run_measure_clustering(config, resources=resources),
                checkpoint_path=checkpoints_path,
                progress_callback=progress_callback,
            )
        relations_checkpoint = resumed_checkpoint(checkpoints, "relations", progress_callback)
        if relations_checkpoint is None:
            relations_checkpoint, _relations_result = _run_stage(
                name="relations",
                config=config,
                checkpoints=checkpoints,
                action=lambda: run_measure_relations(config, resources=resources),
                checkpoint_path=checkpoints_path,
                progress_callback=progress_callback,
            )
        search_checkpoint = resumed_checkpoint(checkpoints, "build-search-index", progress_callback)
        if search_checkpoint is None:
            search_checkpoint, _search_result = _run_stage(
                name="build-search-index",
                config=config,
                checkpoints=checkpoints,
                action=lambda: _search_stage(config, resources),
                optional=True,
                checkpoint_path=checkpoints_path,
                progress_callback=progress_callback,
            )
        if search_checkpoint["status"] == "succeeded":
            retrieval_checkpoint = resumed_checkpoint(
                checkpoints,
                "evaluate-retrieval",
                progress_callback,
            )
            if retrieval_checkpoint is None:
                _retrieval_checkpoint, _retrieval_result = _run_stage(
                    name="evaluate-retrieval",
                    config=config,
                    checkpoints=checkpoints,
                    action=lambda: _retrieval_stage(config),
                    optional=True,
                    checkpoint_path=checkpoints_path,
                    progress_callback=progress_callback,
                )
        else:
            retrieval_checkpoint = resumed_checkpoint(
                checkpoints,
                "evaluate-retrieval",
                progress_callback,
            )
            if retrieval_checkpoint is None:
                skipped_checkpoint(
                    name="evaluate-retrieval",
                    config=config,
                    checkpoints=checkpoints,
                    reason="full semantic search index was not completed",
                    optional=True,
                    checkpoint_path=checkpoints_path,
                    progress_callback=progress_callback,
                )
        _ = cluster_checkpoint, relations_checkpoint
    else:
        for stage_name in (
            "cluster-measures",
            "relations",
            "build-search-index",
            "evaluate-retrieval",
        ):
            existing = resumed_checkpoint(checkpoints, stage_name, progress_callback)
            if existing is not None:
                continue
            skipped_checkpoint(
                name=stage_name,
                config=config,
                checkpoints=checkpoints,
                reason="classification did not complete in this run",
                optional=stage_name in {"build-search-index", "evaluate-retrieval"},
                checkpoint_path=checkpoints_path,
                progress_callback=progress_callback,
            )

    checkpoints_path = _write_checkpoints(checkpoints, checkpoints_path)
    measurements_path = _write_resource_measurements(
        checkpoints,
        measurements_path,
    )
    report_path = _write_scalability_report(
        run_id=actual_run_id,
        checkpoints=checkpoints,
        path=config.paths.reports_dir / "full_corpus_scalability.md",
    )
    deliverables = {
        "stage_checkpoints": checkpoints_path,
        "resource_measurements": measurements_path,
        "scalability_report": report_path,
    }
    full_manifest_path = _write_full_manifest(
        config=config,
        run_id=actual_run_id,
        manifest_path=output_dir / "run_manifest.json",
        checkpoints=checkpoints,
        deliverables=deliverables,
    )
    deliverables["run_manifest"] = full_manifest_path
    completed = complete_manifest(
        manifest,
        artifacts=tuple(str(path) for path in deliverables.values()),
    )
    manifest_index_path = write_manifest(
        completed,
        config.paths.outputs_dir / "manifests" / f"{actual_run_id}_run_all.json",
    )
    return {
        "run_id": actual_run_id,
        "run_manifest": str(full_manifest_path),
        "manifest_index": str(manifest_index_path),
        "stage_checkpoints": str(checkpoints_path),
        "resource_measurements": str(measurements_path),
        "scalability_report": str(report_path),
        "stage_status_counts": {
            status: sum(1 for checkpoint in checkpoints if checkpoint["status"] == status)
            for status in ("succeeded", "failed", "blocked", "skipped")
        },
    }
