from pathlib import Path

import pytest

from statvocab.artifact_validation import validate_artifacts
from statvocab.config import AppConfig, load_config
from statvocab.full_corpus import (
    _run_stage,
    artifact_info,
    disk_preflight,
    run_full_corpus,
)


def _full_config(tmp_path: Path, *, expected_count: int = 1) -> AppConfig:
    base = load_config("configs/full.yaml")
    paths = base.paths.model_copy(
        update={
            "data_dir": tmp_path / "data",
            "raw_dir": tmp_path / "data" / "raw",
            "external_dir": tmp_path / "data" / "external",
            "processed_dir": tmp_path / "data" / "processed",
            "outputs_dir": tmp_path / "outputs",
            "reports_dir": tmp_path / "report",
            "cache_dir": tmp_path / ".cache",
        }
    )
    corpus = base.corpus.model_copy(
        update={
            "archive_name": "eurostat_7605_tables.tgz",
            "expected_table_count": expected_count,
        }
    )
    evaluation = base.evaluation.model_copy(update={"extraction_gold_dir": tmp_path / "gold"})
    return base.model_copy(update={"paths": paths, "corpus": corpus, "evaluation": evaluation})


def _touch(path: Path, text: str = "ok\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_disk_preflight_allows_exact_extracted_cache_without_archive(tmp_path: Path) -> None:
    config = _full_config(tmp_path, expected_count=2)
    raw_cache = config.paths.raw_dir / "eurostat_7605_tables"
    _touch(raw_cache / "a.csv")
    _touch(raw_cache / "b.csv")

    payload = disk_preflight(config)

    assert payload["archive_exists"] is False
    assert payload["extracted_csv_count"] == 2
    assert payload["warnings"]


def test_artifact_info_records_size_and_checksum_for_small_file(tmp_path: Path) -> None:
    path = _touch(tmp_path / "artifact.txt", "artifact\n")

    info = artifact_info(path)

    assert info["exists"] is True
    assert info["size_bytes"] == path.stat().st_size
    assert info["md5"]


def test_run_stage_records_failure_checkpoint(tmp_path: Path) -> None:
    config = _full_config(tmp_path)
    checkpoints: list[dict[str, object]] = []

    checkpoint, result = _run_stage(
        name="failing-stage",
        config=config,
        checkpoints=checkpoints,
        action=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert result is None
    assert checkpoint["status"] == "failed"
    assert checkpoint["failure_message"] == "boom"
    assert checkpoints == [checkpoint]


def test_full_corpus_runner_records_semantic_search_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import statvocab.full_corpus as full_corpus

    config = _full_config(tmp_path)

    def fake_acquire_stage(config: AppConfig) -> dict[str, object]:
        path = _touch(config.paths.processed_dir / "resource_manifest.json", "[]\n")
        return {
            "artifacts": {"resource_manifest": path},
            "diagnostics": {"run_id": "run_acquire"},
            "resources": (),
        }

    def fake_tuple_stage(
        config: AppConfig,
        *args: object,
        **kwargs: object,
    ) -> tuple[dict[str, Path], Path, dict[str, object]]:
        _ = args, kwargs
        name = "stage"
        artifact_count = len(list(config.paths.processed_dir.glob("*")))
        artifact = _touch(config.paths.processed_dir / f"{name}_{artifact_count}.txt")
        manifest = _touch(config.paths.outputs_dir / "manifests" / f"run_{artifact.stem}.json")
        return {"artifact": artifact}, manifest, {"run_id": f"run_{artifact.stem}"}

    def fake_search_stage(config: AppConfig, resources: object) -> dict[str, object]:
        _ = resources
        lexical = _touch(config.paths.outputs_dir / "search" / "run_test" / "lexical.sqlite")
        return {
            "status": "blocked",
            "failure_message": "semantic search blocked",
            "artifacts": {"lexical_index": lexical},
            "diagnostics": {"run_id": "run_search", "semantic_status": "blocked"},
        }

    monkeypatch.setattr(full_corpus, "disk_preflight", lambda config: {"ok": True})
    monkeypatch.setattr(full_corpus, "_acquire_stage", fake_acquire_stage)
    monkeypatch.setattr(full_corpus, "run_ingestion", fake_tuple_stage)
    monkeypatch.setattr(full_corpus, "run_extraction", fake_tuple_stage)
    monkeypatch.setattr(full_corpus, "run_classification", fake_tuple_stage)
    monkeypatch.setattr(full_corpus, "run_measure_clustering", fake_tuple_stage)
    monkeypatch.setattr(full_corpus, "run_measure_relations", fake_tuple_stage)
    monkeypatch.setattr(full_corpus, "_search_stage", fake_search_stage)

    payload = run_full_corpus(config)

    assert payload["stage_status_counts"]["blocked"] == 1
    assert payload["stage_status_counts"]["skipped"] == 1
    checkpoints_path = Path(str(payload["stage_checkpoints"]))
    assert "semantic search blocked" in checkpoints_path.read_text(encoding="utf-8")


def test_full_corpus_validation_accepts_incomplete_stages_with_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import statvocab.full_corpus as full_corpus

    config = _full_config(tmp_path)
    monkeypatch.setattr(full_corpus, "disk_preflight", lambda config: {"ok": True})
    monkeypatch.setattr(
        full_corpus,
        "_acquire_stage",
        lambda config: (_ for _ in ()).throw(RuntimeError("resource missing")),
    )

    payload = run_full_corpus(config)
    validation = validate_artifacts(config, run_id=str(payload["run_id"]))

    assert validation["validated"] is True
    assert validation["pipeline_stage"] == "run-all"
    assert validation["incomplete_stages"]
