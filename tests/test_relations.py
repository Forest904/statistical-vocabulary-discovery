import csv
import json
from pathlib import Path
from typing import Any

from statvocab.classification_features import write_parquet_rows
from statvocab.config import AppConfig, load_config
from statvocab.contracts import RelationType
from statvocab.relations import (
    _Candidate,
    _dedupe,
    run_measure_relations,
)
from statvocab.relations_evaluate import evaluate_relations


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


def _write_measures(config: AppConfig, rows: list[dict[str, Any]]) -> None:
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
        for row in rows:
            writer.writerow(
                {
                    "term_id": row["term_id"],
                    "canonical_term": row["canonical_term"],
                    "category": "measure",
                    "confidence": "1.0",
                    "variant": "test",
                    "protected": "true",
                    "evidence": "test",
                    "occurrence_ids_json": json.dumps([f"occ_{row['term_id']}"]),
                    "run_id": "run_test",
                }
            )


def _write_embeddings(config: AppConfig, rows: list[dict[str, Any]]) -> None:
    write_parquet_rows(
        [
            {
                "term_id": row["term_id"],
                "canonical_term": row["canonical_term"],
                "model": config.classification.pearl_model,
                "revision": config.classification.pearl_revision,
                "embedding": row["embedding"],
            }
            for row in rows
        ],
        config.paths.processed_dir / "term_embeddings.parquet",
    )


def _relation_rows(config: AppConfig) -> list[dict[str, str]]:
    with (config.paths.outputs_dir / "measure_relations.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def test_measure_relations_export_grounded_candidates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rows = [
        {"term_id": "term_population", "canonical_term": "population", "embedding": [1.0, 0.0]},
        {
            "term_id": "term_urban_population",
            "canonical_term": "urban population",
            "embedding": [0.95, 0.05],
        },
        {
            "term_id": "term_price_index",
            "canonical_term": "consumer price index",
            "embedding": [0.0, 1.0],
        },
        {
            "term_id": "term_retail_index",
            "canonical_term": "retail price index",
            "embedding": [0.0, 0.98],
        },
        {
            "term_id": "term_gdp",
            "canonical_term": "gross domestic product",
            "embedding": [0.4, 0.6],
        },
        {
            "term_id": "term_gdp_variant",
            "canonical_term": "domestic gross product",
            "embedding": [0.4, 0.6],
        },
    ]
    _write_measures(config, rows)
    _write_embeddings(config, rows)

    artifacts, _manifest_path, diagnostics = run_measure_relations(config)

    exported = _relation_rows(config)
    assert artifacts["measure_relations"].exists()
    assert diagnostics["accepted_count"] >= 3
    assert all(row["evidence_ids_json"] and row["confidence"] for row in exported)
    triples = {
        (row["source_term_id"], row["target_term_id"], row["relation_type"])
        for row in exported
    }
    assert triples >= {
        ("term_population", "term_urban_population", "broader_than"),
        ("term_price_index", "term_retail_index", "related_to"),
    }
    assert (
        frozenset(("term_gdp", "term_gdp_variant")),
        "variant_of",
    ) in {
        (frozenset((row["source_term_id"], row["target_term_id"])), row["relation_type"])
        for row in exported
    }


def test_relations_evaluation_reports_artifact_integrity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_measures(
        config,
        [{"term_id": "term_a", "canonical_term": "population", "embedding": [1.0, 0.0]}],
    )
    with (config.paths.outputs_dir / "measure_relations.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "relation_id",
                "source_term_id",
                "source_term",
                "target_term_id",
                "target_term",
                "relation_type",
                "confidence",
                "evidence_ids_json",
                "evidence",
                "generation_methods_json",
                "run_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "relation_id": "relation_a",
                "source_term_id": "term_a",
                "source_term": "population",
                "target_term_id": "term_unknown",
                "target_term": "unknown",
                "relation_type": "related_to",
                "confidence": "0.75",
                "evidence_ids_json": '["occ_a"]',
                "evidence": "{}",
                "generation_methods_json": '["test"]',
                "run_id": "run_test",
            }
        )

    _metrics_path, payload = evaluate_relations(config)

    assert payload["validation"]["passed"] is False
    assert "unknown measures" in payload["validation"]["failures"][0]


def test_relation_dedupe_rejects_self_and_duplicate_pairs() -> None:
    candidates = [
        _Candidate(
            source_term_id="term_a",
            target_term_id="term_a",
            relation_type=RelationType.RELATED_TO,
            confidence=0.9,
            evidence_ids=("occ_a",),
            evidence={},
            methods=("test",),
        ),
        _Candidate(
            source_term_id="term_a",
            target_term_id="term_b",
            relation_type=RelationType.RELATED_TO,
            confidence=0.7,
            evidence_ids=("occ_a", "occ_b"),
            evidence={},
            methods=("test",),
        ),
        _Candidate(
            source_term_id="term_b",
            target_term_id="term_a",
            relation_type=RelationType.RELATED_TO,
            confidence=0.8,
            evidence_ids=("occ_a", "occ_b"),
            evidence={},
            methods=("test",),
        ),
    ]

    accepted, rejections = _dedupe(candidates, {"term_a", "term_b"})

    assert len(accepted) == 1
    assert accepted[0].source_term_id == "term_b"
    assert rejections[0] == "self_relation"
    assert rejections[1] == "duplicate_lower_confidence"


def test_empty_measure_input_writes_empty_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_measures(config, [])

    artifacts, _manifest_path, diagnostics = run_measure_relations(config)
    rows = _relation_rows(config)

    assert artifacts["measure_relations"].exists()
    assert diagnostics["candidate_count"] == 0
    assert rows == []
