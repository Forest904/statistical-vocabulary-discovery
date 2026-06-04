import csv
from pathlib import Path
from typing import Any

import pytest

from statvocab.classification_features import write_parquet_rows
from statvocab.cluster_measures import CONTROLLED_DOMAINS, run_measure_clustering
from statvocab.clustering_evaluate import evaluate_clustering
from statvocab.config import AppConfig, load_config


def _config(tmp_path: Path) -> AppConfig:
    base = load_config("configs/evaluation.yaml")
    paths = base.paths.model_copy(
        update={
            "processed_dir": tmp_path / "processed",
            "outputs_dir": tmp_path / "outputs",
            "reports_dir": tmp_path / "report",
        }
    )
    return base.model_copy(update={"paths": paths})


def _write_measures(config: AppConfig) -> None:
    config.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    with (config.paths.outputs_dir / "measures.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "term_id",
                "canonical_term",
                "category",
                "confidence",
                "variant",
                "protected",
                "evidence",
                "occurrence_ids_json",
                "run_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "term_id": "term_a",
                    "canonical_term": "gross domestic product",
                    "category": "measure",
                    "confidence": "1.0",
                    "variant": "test",
                    "protected": "true",
                    "evidence": "test",
                    "occurrence_ids_json": "[]",
                    "run_id": "run_test",
                },
                {
                    "term_id": "term_b",
                    "canonical_term": "consumer price index",
                    "category": "measure",
                    "confidence": "1.0",
                    "variant": "test",
                    "protected": "true",
                    "evidence": "test",
                    "occurrence_ids_json": "[]",
                    "run_id": "run_test",
                },
                {
                    "term_id": "term_c",
                    "canonical_term": "urban population",
                    "category": "measure",
                    "confidence": "1.0",
                    "variant": "test",
                    "protected": "true",
                    "evidence": "test",
                    "occurrence_ids_json": "[]",
                    "run_id": "run_test",
                },
            ]
        )


def _write_embeddings(config: AppConfig) -> None:
    write_parquet_rows(
        [
            {
                "term_id": "term_a",
                "canonical_term": "gross domestic product",
                "model": config.classification.pearl_model,
                "revision": config.classification.pearl_revision,
                "embedding": [1.0, 0.0],
            },
            {
                "term_id": "term_b",
                "canonical_term": "consumer price index",
                "model": config.classification.pearl_model,
                "revision": config.classification.pearl_revision,
                "embedding": [0.9, 0.1],
            },
            {
                "term_id": "term_c",
                "canonical_term": "urban population",
                "model": config.classification.pearl_model,
                "revision": config.classification.pearl_revision,
                "embedding": [-1.0, 0.0],
            },
        ],
        config.paths.processed_dir / "term_embeddings.parquet",
    )


class _FakeNumpy:
    def array(self, values: list[list[float]], dtype: Any = float) -> list[list[float]]:
        _ = dtype
        return values


class _FakeHdbscanModel:
    labels_ = [0, 0, -1]
    probabilities_ = [0.95, 0.9, 0.0]

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit(self, matrix: list[list[float]]) -> "_FakeHdbscanModel":
        assert matrix
        return self


class _FakeHdbscanModule:
    HDBSCAN = _FakeHdbscanModel


class _FakeAgglomerative:
    labels_ = [0, 0, 1]

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit(self, matrix: list[list[float]]) -> "_FakeAgglomerative":
        assert matrix
        return self


def test_measure_clustering_exports_valid_memberships(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _write_measures(config)
    _write_embeddings(config)

    import statvocab.cluster_measures as cluster_measures

    monkeypatch.setattr(
        cluster_measures,
        "_import_clustering_dependencies",
        lambda: (_FakeNumpy(), _FakeHdbscanModule(), _FakeAgglomerative, lambda *_: 0.42),
    )
    monkeypatch.setattr(
        cluster_measures,
        "_domain_embeddings",
        lambda config, measure_count: {
            "economy and finance": [1.0, 0.0],
            "labour market": [0.0, 1.0],
        },
    )

    artifacts, _manifest_path, diagnostics = run_measure_clustering(config)

    rows = list(csv.DictReader((config.paths.outputs_dir / "measure_clusters.csv").open()))
    measure_ids = {"term_a", "term_b", "term_c"}
    assert artifacts["baseline_assignments"].exists()
    assert diagnostics["measure_count"] == 3
    assert {row["term_id"] for row in rows} == measure_ids
    assert len(rows) == len(measure_ids)
    assert any(row["cluster_id"] == "unclustered" for row in rows)
    assert {row["domain"] for row in rows} <= set(CONTROLLED_DOMAINS)
    assert all(
        row["cluster_id"] != "unclustered"
        for row in rows
        if row["is_representative"] == "true"
    )


def test_clustering_evaluation_reports_artifact_integrity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    with (config.paths.outputs_dir / "measure_clusters.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "term_id",
                "term",
                "cluster_id",
                "domain",
                "membership_probability",
                "is_representative",
                "labeling_method",
                "evidence",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "term_id": "term_a",
                "term": "gross domestic product",
                "cluster_id": "cluster_a",
                "domain": "economy and finance",
                "membership_probability": "0.9",
                "is_representative": "true",
                "labeling_method": "embedding_threshold",
                "evidence": "{}",
            }
        )

    metrics_path, payload = evaluate_clustering(config)

    assert metrics_path.exists()
    assert payload["artifact_status"] == "available"
    assert payload["validation"]["passed"] is True
    assert payload["coverage"] == 1.0


def test_missing_clustering_dependencies_error_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    import statvocab.cluster_measures as cluster_measures

    def fail_import(name: str) -> Any:
        if name == "hdbscan":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr(cluster_measures, "import_module", fail_import)

    with pytest.raises(RuntimeError, match=r"pip install -e \.\[ml\]"):
        cluster_measures._import_clustering_dependencies()
