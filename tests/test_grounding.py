import csv
from pathlib import Path

import pyarrow.parquet as pq

from statvocab.artifact_validation import validate_artifacts
from statvocab.classification import run_classification
from statvocab.classification_evaluate import evaluate_classification
from statvocab.classification_features import ensure_gold_templates, feature_rows, read_csv_rows
from statvocab.config import AppConfig, load_config
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
    evaluation = base.evaluation.model_copy(update={"extraction_gold_dir": tmp_path / "gold"})
    return base.model_copy(
        update={"paths": paths, "extraction": extraction, "evaluation": evaluation}
    )


def test_feature_generation_exposes_structural_evidence(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    run_ingestion(config)
    run_extraction(config)

    rows = feature_rows(config)

    assert rows
    assert {"term_id", "stratum", "has_title_evidence", "has_metadata_value_evidence"} <= set(
        rows[0]
    )


def test_rule_only_classification_exports_grounded_partition(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    run_ingestion(config)
    run_extraction(config)

    artifacts, _manifest_path, diagnostics = run_classification(config, variant="rule-only")

    predictions = pq.read_table(artifacts["predictions"]).to_pylist()
    vocabulary = pq.read_table(config.paths.processed_dir / "vocabulary.parquet").to_pylist()
    validation = validate_artifacts(config, run_id=diagnostics["run_id"])

    assert len(predictions) == len(vocabulary)
    assert validation["validated"] is True
    assert validation["term_count"] == len(vocabulary)
    assert (config.paths.outputs_dir / "measures.csv").exists()
    assert (config.paths.outputs_dir / "other_ambiguous.csv").exists()


def test_classification_evaluation_is_pending_without_gold_labels(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    run_ingestion(config)
    run_extraction(config)

    metrics_path, payload = evaluate_classification(config)

    assert metrics_path.exists()
    assert payload["gold_status"] == "pending"
    assert payload["metrics_status"] == "pending_predictions"


def test_gold_template_creation_preserves_completed_relabels(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    run_ingestion(config)
    run_extraction(config)
    _sample_path, _labels_path, relabel_path = ensure_gold_templates(config)

    rows = read_csv_rows(relabel_path)
    rows[0]["category"] = "measure"
    rows[0]["annotator_id"] = "test"
    rows[0]["notes"] = "completed duplicate label"
    with relabel_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ensure_gold_templates(config)

    preserved = read_csv_rows(relabel_path)
    assert preserved[0]["category"] == "measure"
    assert preserved[0]["notes"] == "completed duplicate label"
