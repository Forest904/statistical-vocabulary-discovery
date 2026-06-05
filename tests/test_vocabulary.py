from pathlib import Path

import pyarrow.parquet as pq

from statvocab.config import AppConfig, load_config
from statvocab.extraction_evaluate import run_extraction_evaluation
from statvocab.ingest import run_ingestion
from statvocab.vocabulary import run_extraction


def _fixture_config(tmp_path: Path) -> AppConfig:
    base = load_config("configs/evaluation.yaml")
    paths = base.paths.model_copy(
        update={
            "processed_dir": tmp_path / "processed",
            "outputs_dir": tmp_path / "outputs",
            "reports_dir": tmp_path / "report",
            "external_dir": tmp_path / "external",
        }
    )
    extraction = base.extraction.model_copy(
        update={
            "nuts_2024_path": tmp_path / "external" / "missing_nuts.csv",
            "eurostat_geo_codelist_path": tmp_path / "external" / "missing_geo.xml",
        }
    )
    return base.model_copy(update={"paths": paths, "extraction": extraction})


def test_fixture_extraction_writes_all_milestone_2_artifacts(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    run_ingestion(config)

    artifacts, _manifest_path, diagnostics = run_extraction(config)

    expected = {
        "table_times",
        "table_strings",
        "table_geographies",
        "title_terms",
        "term_occurrences",
        "table_vocabulary",
        "vocabulary",
        "diagnostics",
    }
    assert expected == set(artifacts)
    assert diagnostics["table_count"] == 4
    assert diagnostics["vocabulary_count"] > 0
    assert all(path.exists() for path in artifacts.values())


def test_fixture_extraction_excludes_observations_and_accepted_geography(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    run_ingestion(config)
    run_extraction(config)

    occurrences = pq.read_table(config.paths.processed_dir / "term_occurrences.parquet").to_pylist()
    vocabulary = pq.read_table(config.paths.processed_dir / "vocabulary.parquet").to_pylist()
    geographies = pq.read_table(
        config.paths.processed_dir / "table_geographies.parquet"
    ).to_pylist()

    keys = {row["matching_key"] for row in occurrences}
    vocabulary_keys = [row["matching_key"] for row in vocabulary]

    assert "10" not in keys
    assert "22e" not in keys
    assert "fr" not in keys
    assert "italy" not in keys
    assert any(row["raw_value"] == "FR" for row in geographies)
    assert len(vocabulary_keys) == len(set(vocabulary_keys))
    assert all(row["table_id"] and row["location"] for row in occurrences)


def test_fixture_extraction_is_deterministic_for_vocabulary_artifacts(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    run_ingestion(config)

    run_extraction(config)
    first = pq.read_table(config.paths.processed_dir / "vocabulary.parquet").to_pylist()
    run_extraction(config)
    second = pq.read_table(config.paths.processed_dir / "vocabulary.parquet").to_pylist()

    assert first == second


def test_partial_extraction_resume_reuses_table_fragments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import statvocab.vocabulary as vocabulary

    config = _fixture_config(tmp_path)
    run_ingestion(config)
    events: list[dict[str, object]] = []

    run_extraction(
        config,
        partial_run_id="run_partial",
        resume_partials=True,
        progress_callback=events.append,
    )
    first = pq.read_table(config.paths.processed_dir / "vocabulary.parquet").to_pylist()
    partial_dir = config.paths.processed_dir / "full_extract_partial" / "run_partial"

    assert (partial_dir / "run_state.json").exists()
    assert (partial_dir / "table_progress.jsonl").exists()
    assert len(list((partial_dir / "tables").glob("*.json"))) == 4
    assert any(event.get("event") == "table_finished" for event in events)

    def fail_if_raw_table_is_reprocessed(*_args, **_kwargs):
        raise AssertionError("resume should reuse partial table fragments")

    monkeypatch.setattr(vocabulary, "_extract_string_occurrences", fail_if_raw_table_is_reprocessed)

    run_extraction(
        config,
        partial_run_id="run_partial",
        resume_partials=True,
        progress_callback=events.append,
    )
    second = pq.read_table(config.paths.processed_dir / "vocabulary.parquet").to_pylist()

    assert first == second
    assert any(event.get("resumed") is True for event in events)


def test_extraction_evaluation_uses_fixture_gold_labels(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    run_ingestion(config)
    run_extraction(config)

    _review_path, metrics_path, payload = run_extraction_evaluation(config)

    assert metrics_path.exists()
    assert payload["gold_status"] == "available"
    assert payload["metrics"]["time"]["f1"] == 1.0


def test_extraction_evaluation_without_gold_labels_is_pending(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    config = config.model_copy(
        update={
            "corpus": config.corpus.model_copy(update={"name": "core"}),
            "evaluation": config.evaluation.model_copy(
                update={"extraction_gold_dir": tmp_path / "gold"}
            ),
        }
    )
    fixture_corpus = load_config("configs/evaluation.yaml").corpus
    run_ingestion(config.model_copy(update={"corpus": fixture_corpus}))

    _review_path, metrics_path, payload = run_extraction_evaluation(config)

    assert metrics_path.exists()
    assert payload["gold_status"] == "pending"
    assert "annotation template" in payload["message"]


def test_core_extraction_evaluation_uses_tracked_gold_dir(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    run_ingestion(config)
    run_extraction(config)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "extraction_gold_labels.csv").write_text(
        "table_id,area,expected_value,expected_present,notes\n"
        "fixture_regular,time,2019,true,fixture-style core gold label\n",
        encoding="utf-8",
    )
    core_config = config.model_copy(
        update={
            "corpus": config.corpus.model_copy(update={"name": "core"}),
            "evaluation": config.evaluation.model_copy(update={"extraction_gold_dir": gold_dir}),
        }
    )

    _review_path, _metrics_path, payload = run_extraction_evaluation(core_config)

    assert payload["gold_labels"] == str(gold_dir / "extraction_gold_labels.csv")
    assert payload["gold_status"] == "available"
    assert payload["metrics"]["time"]["tp"] == 1
