from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from statvocab.classification import CSV_BY_CATEGORY
from statvocab.classification_features import read_parquet_rows, write_parquet_rows
from statvocab.config import AppConfig, load_config
from statvocab.knowledge_graph import run_knowledge_graph, validate_graph_rows


def _config(tmp_path: Path) -> AppConfig:
    base = load_config("configs/evaluation.yaml")
    paths = base.paths.model_copy(
        update={
            "processed_dir": tmp_path / "processed",
            "outputs_dir": tmp_path / "outputs",
            "reports_dir": tmp_path / "report",
        }
    )
    knowledge_graph = base.knowledge_graph.model_copy(
        update={"min_table_relation_weight": 0.15, "max_table_edges_per_table": 1}
    )
    return base.model_copy(update={"paths": paths, "knowledge_graph": knowledge_graph})


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_category_outputs(config: AppConfig) -> None:
    fields = [
        "term_id",
        "canonical_term",
        "category",
        "confidence",
        "variant",
        "protected",
        "evidence",
        "occurrence_ids_json",
        "run_id",
    ]
    rows_by_file = {filename: [] for filename in CSV_BY_CATEGORY.values()}
    rows_by_file["measures.csv"] = [
        {
            "term_id": "term_measure_a",
            "canonical_term": "Employment",
            "category": "measure",
            "confidence": "0.9",
            "variant": "test",
            "protected": "true",
            "evidence": "title",
            "occurrence_ids_json": json.dumps(["occ_a"]),
            "run_id": "run_test",
        },
        {
            "term_id": "term_measure_b",
            "canonical_term": "Employment rate",
            "category": "measure",
            "confidence": "0.88",
            "variant": "test",
            "protected": "true",
            "evidence": "title",
            "occurrence_ids_json": json.dumps(["occ_b"]),
            "run_id": "run_test",
        },
    ]
    rows_by_file["dimension_names.csv"] = [
        {
            "term_id": "term_dimension",
            "canonical_term": "Sex",
            "category": "dimension_name",
            "confidence": "0.8",
            "variant": "test",
            "protected": "true",
            "evidence": "header",
            "occurrence_ids_json": json.dumps(["occ_c"]),
            "run_id": "run_test",
        }
    ]
    for filename, rows in rows_by_file.items():
        _write_csv(config.paths.outputs_dir / filename, rows, fields)


def _write_base_artifacts(config: AppConfig) -> None:
    write_parquet_rows(
        [
            {
                "table_id": "table_1",
                "filename": "one.csv",
                "title": "Employment by sex",
                "source_url": "https://example.test/1",
                "parse_status": "parsed",
                "row_count": 10,
                "observation_count": 20,
            },
            {
                "table_id": "table_2",
                "filename": "two.csv",
                "title": "Employment rate by sex",
                "source_url": "https://example.test/2",
                "parse_status": "parsed",
                "row_count": 12,
                "observation_count": 24,
            },
            {
                "table_id": "table_3",
                "filename": "three.csv",
                "title": "Other labour table",
                "source_url": "https://example.test/3",
                "parse_status": "parsed",
                "row_count": 8,
                "observation_count": 18,
            },
        ],
        config.paths.processed_dir / "tables.parquet",
    )
    write_parquet_rows(
        [
            {
                "term_id": "term_measure_a",
                "canonical_term": "Employment",
                "matching_key": "employment",
                "table_count": 2,
                "occurrence_count": 3,
                "occurrence_ids_json": json.dumps(["occ_a"]),
                "run_id": "run_test",
            },
            {
                "term_id": "term_measure_b",
                "canonical_term": "Employment rate",
                "matching_key": "employment rate",
                "table_count": 1,
                "occurrence_count": 1,
                "occurrence_ids_json": json.dumps(["occ_b"]),
                "run_id": "run_test",
            },
            {
                "term_id": "term_dimension",
                "canonical_term": "Sex",
                "matching_key": "sex",
                "table_count": 2,
                "occurrence_count": 2,
                "occurrence_ids_json": json.dumps(["occ_c"]),
                "run_id": "run_test",
            },
        ],
        config.paths.processed_dir / "vocabulary.parquet",
    )
    write_parquet_rows(
        [
            {
                "table_id": "table_1",
                "term_id": "term_measure_a",
                "canonical_term": "Employment",
                "occurrence_count": 1,
                "occurrence_ids_json": json.dumps(["occ_a1"]),
                "source_areas_json": json.dumps(["title"]),
            },
            {
                "table_id": "table_1",
                "term_id": "term_dimension",
                "canonical_term": "Sex",
                "occurrence_count": 1,
                "occurrence_ids_json": json.dumps(["occ_c1"]),
                "source_areas_json": json.dumps(["metadata_column_name"]),
            },
            {
                "table_id": "table_2",
                "term_id": "term_measure_b",
                "canonical_term": "Employment rate",
                "occurrence_count": 1,
                "occurrence_ids_json": json.dumps(["occ_b1"]),
                "source_areas_json": json.dumps(["title"]),
            },
            {
                "table_id": "table_3",
                "term_id": "term_measure_a",
                "canonical_term": "Employment",
                "occurrence_count": 2,
                "occurrence_ids_json": json.dumps(["occ_a2", "occ_a3"]),
                "source_areas_json": json.dumps(["metadata_value"]),
            },
        ],
        config.paths.processed_dir / "table_vocabulary.parquet",
    )
    _write_category_outputs(config)
    _write_csv(
        config.paths.outputs_dir / "measure_clusters.csv",
        [
            {
                "term_id": "term_measure_a",
                "term": "Employment",
                "cluster_id": "cluster_labour",
                "domain": "labour market",
                "membership_probability": "0.92",
                "is_representative": "true",
                "labeling_method": "test",
                "evidence": "{}",
            },
            {
                "term_id": "term_measure_b",
                "term": "Employment rate",
                "cluster_id": "cluster_labour",
                "domain": "labour market",
                "membership_probability": "0.91",
                "is_representative": "false",
                "labeling_method": "test",
                "evidence": "{}",
            },
        ],
        [
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
    _write_csv(
        config.paths.outputs_dir / "measure_relations.csv",
        [
            {
                "relation_id": "relation_test",
                "source_term_id": "term_measure_a",
                "source_term": "Employment",
                "target_term_id": "term_measure_b",
                "target_term": "Employment rate",
                "relation_type": "related_to",
                "confidence": "0.8",
                "evidence_ids_json": json.dumps(["occ_a", "occ_b"]),
                "evidence": json.dumps({"embedding_similarity": 0.9}),
                "generation_methods_json": json.dumps(["test"]),
                "run_id": "run_test",
            }
        ],
        [
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


def test_knowledge_graph_builds_expected_semantic_layers(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_base_artifacts(config)

    artifacts, _manifest_path, diagnostics = run_knowledge_graph(config)

    nodes = read_parquet_rows(artifacts["nodes"])
    edges = read_parquet_rows(artifacts["edges"])
    node_types = {row["node_type"] for row in nodes}
    edge_types = {row["edge_type"] for row in edges}

    assert {"table", "term", "category", "cluster", "domain"} <= node_types
    assert {
        "table_contains_term",
        "term_classified_as",
        "measure_member_of_cluster",
        "cluster_belongs_to_domain",
        "related_to",
        "table_related_to_table",
    } <= edge_types
    assert diagnostics["node_count"] == len(nodes)
    assert diagnostics["edge_count"] == len(edges)


def test_graph_validation_rejects_unknown_node_references() -> None:
    with pytest.raises(ValueError, match="unknown nodes"):
        validate_graph_rows(
            [{"node_id": "table_1", "node_type": "table"}],
            [
                {
                    "edge_id": "edge_1",
                    "source_id": "table_1",
                    "target_id": "missing",
                    "edge_type": "table_contains_term",
                }
            ],
        )


def test_graph_validation_rejects_self_table_relations() -> None:
    with pytest.raises(ValueError, match="self edges"):
        validate_graph_rows(
            [{"node_id": "table_1", "node_type": "table"}],
            [
                {
                    "edge_id": "edge_1",
                    "source_id": "table_1",
                    "target_id": "table_1",
                    "edge_type": "table_related_to_table",
                }
            ],
        )


def test_table_relation_edges_obey_per_table_limit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_base_artifacts(config)

    artifacts, _manifest_path, _diagnostics = run_knowledge_graph(config)
    table_edges = [
        row
        for row in read_parquet_rows(artifacts["edges"])
        if row["edge_type"] == "table_related_to_table"
    ]
    degree: dict[str, int] = {}
    for edge in table_edges:
        degree[str(edge["source_id"])] = degree.get(str(edge["source_id"]), 0) + 1
        degree[str(edge["target_id"])] = degree.get(str(edge["target_id"]), 0) + 1

    assert table_edges
    assert max(degree.values()) <= 1
