import json
from pathlib import Path

from statvocab.classification_features import read_csv_rows, write_csv_rows, write_parquet_rows
from statvocab.config import AppConfig, load_config
from statvocab.human_loop import (
    append_review_answer,
    compile_human_labels,
    generate_review_tasks,
    next_review_task,
    review_stats,
)


def _config(tmp_path: Path) -> AppConfig:
    base = load_config("configs/evaluation.yaml")
    return base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={
                    "data_dir": tmp_path / "data",
                    "processed_dir": tmp_path / "data" / "processed",
                    "outputs_dir": tmp_path / "outputs",
                    "reports_dir": tmp_path / "report",
                }
            ),
            "evaluation": base.evaluation.model_copy(
                update={"extraction_gold_dir": tmp_path / "data" / "gold"}
            ),
        }
    )


def _write_fixture_artifacts(config: AppConfig) -> None:
    processed = config.paths.processed_dir
    outputs = config.paths.outputs_dir
    gold = config.evaluation.extraction_gold_dir
    gold.mkdir(parents=True)
    write_parquet_rows(
        [
            {
                "table_id": "table_a",
                "title": "Population density by region",
                "parse_status": "parsed",
            }
        ],
        processed / "tables.parquet",
    )
    write_parquet_rows(
        [
            {
                "term_id": "term_population",
                "canonical_term": "Population density",
                "matching_key": "population density",
                "occurrence_count": 3,
                "table_count": 1,
                "occurrence_ids_json": json.dumps(["occ_a"]),
                "role_summary_json": "{}",
                "feature_summary_json": "{}",
                "run_id": "run_test",
            },
            {
                "term_id": "term_region",
                "canonical_term": "Region",
                "matching_key": "region",
                "occurrence_count": 2,
                "table_count": 1,
                "occurrence_ids_json": json.dumps(["occ_b"]),
                "role_summary_json": "{}",
                "feature_summary_json": "{}",
                "run_id": "run_test",
            },
        ],
        processed / "vocabulary.parquet",
    )
    write_parquet_rows(
        [
            {
                "term_id": "term_population",
                "canonical_term": "Population density",
                "matching_key": "population density",
                "occurrence_count": 3,
                "table_count": 1,
                "occurrence_ids_json": json.dumps(["occ_a"]),
                "source_roles": "title_keyphrase",
                "has_title_evidence": True,
                "has_metadata_value_evidence": False,
                "has_unit_word": False,
                "has_conflicting_roles": False,
                "is_short_code": False,
            },
            {
                "term_id": "term_region",
                "canonical_term": "Region",
                "matching_key": "region",
                "occurrence_count": 2,
                "table_count": 1,
                "occurrence_ids_json": json.dumps(["occ_b"]),
                "source_roles": "metadata_value",
                "has_title_evidence": False,
                "has_metadata_value_evidence": True,
                "has_unit_word": False,
                "has_conflicting_roles": False,
                "is_short_code": False,
            },
        ],
        processed / "classification_features.parquet",
    )
    write_parquet_rows(
        [
            {
                "term_id": "term_population",
                "occurrence_id": "occ_a",
                "table_id": "table_a",
                "source_area": "title_keyphrase",
                "metadata_column": None,
                "location": "title",
            }
        ],
        processed / "term_occurrences.parquet",
    )
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
    for name in (
        "measures.csv",
        "dimension_names.csv",
        "dimension_values.csv",
        "units.csv",
        "other_ambiguous.csv",
    ):
        rows = []
        if name == "other_ambiguous.csv":
            rows = [
                {
                    "term_id": "term_population",
                    "canonical_term": "Population density",
                    "category": "other_ambiguous",
                    "confidence": "0.61",
                    "variant": "test",
                    "protected": "false",
                    "evidence": "test evidence",
                    "occurrence_ids_json": json.dumps(["occ_a"]),
                    "run_id": "run_test",
                }
            ]
        write_csv_rows(rows, outputs / name, fields)
    write_csv_rows(
        [
            {
                "source_term_id": "term_population",
                "source_term": "Population density",
                "target_term_id": "term_region",
                "target_term": "Region",
                "relation_type": "related_to",
                "confidence": "0.7",
            }
        ],
        outputs / "measure_relations.csv",
        [
            "source_term_id",
            "source_term",
            "target_term_id",
            "target_term",
            "relation_type",
            "confidence",
        ],
    )
    write_csv_rows([], gold / "vocabulary_gold_labels.csv", [
        "term_id",
        "canonical_term",
        "matching_key",
        "split",
        "stratum",
        "table_count",
        "occurrence_count",
        "role_summary_json",
        "feature_summary_json",
        "category",
        "annotator_id",
        "notes",
    ])


def test_human_loop_generates_tasks_and_skips_answered(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_fixture_artifacts(config)

    _path, stats = generate_review_tasks(config)
    task = next_review_task(config, mode="term_classification")
    assert task is not None
    append_review_answer(config, task_id=task["task_id"], answer="measure")

    assert stats["task_count"] > 0
    follow_up = next_review_task(config, mode="term_classification")
    assert follow_up is None or follow_up["hidden_qc"] is True


def test_compile_maps_answers_and_ignores_unsure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_fixture_artifacts(config)
    generate_review_tasks(config)
    task = next_review_task(config, mode="term_classification")
    assert task is not None
    append_review_answer(config, task_id=task["task_id"], answer="unsure")
    append_review_answer(config, task_id=task["task_id"], answer="measure")

    _metrics_path, payload = compile_human_labels(config)

    labels = config.evaluation.extraction_gold_dir / "human_loop_classification_labels.csv"
    assert labels.exists()
    assert payload["compiled_classification_label_count"] == 1
    assert payload["classification_label_splits"]
    assert review_stats(config)["event_count"] == 2


def test_compile_preserves_locked_splits(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_fixture_artifacts(config)
    generate_review_tasks(config)
    task = next_review_task(config, mode="term_classification")
    assert task is not None
    append_review_answer(config, task_id=task["task_id"], answer="measure")
    compile_human_labels(config)
    first = (
        config.evaluation.extraction_gold_dir / "human_loop_classification_labels.csv"
    ).read_text(encoding="utf-8")

    compile_human_labels(config)
    second = (
        config.evaluation.extraction_gold_dir / "human_loop_classification_labels.csv"
    ).read_text(encoding="utf-8")

    assert first == second


def test_compile_deduplicates_multiple_task_answers_for_same_term(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_fixture_artifacts(config)
    generate_review_tasks(config)
    classification_task = next_review_task(config, mode="term_classification")
    measure_task = next_review_task(config, mode="measure_vs_breakdown")
    assert classification_task is not None
    assert measure_task is not None
    assert classification_task["term_id"] == measure_task["term_id"]

    append_review_answer(config, task_id=classification_task["task_id"], answer="measure")
    append_review_answer(config, task_id=measure_task["task_id"], answer="dimension_value")
    _metrics_path, payload = compile_human_labels(config)

    labels = read_csv_rows(
        config.evaluation.extraction_gold_dir / "human_loop_classification_labels.csv"
    )
    assert payload["compiled_classification_label_count"] == 1
    assert [row["term_id"] for row in labels] == ["term_population"]
    assert labels[0]["category"] == "dimension_value"


def test_next_task_skips_classification_siblings_for_answered_term(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_fixture_artifacts(config)
    generate_review_tasks(config)
    task = next_review_task(config, mode="term_classification")
    assert task is not None
    append_review_answer(config, task_id=task["task_id"], answer="measure")

    assert next_review_task(config, mode="measure_vs_breakdown") is None
    assert review_stats(config)["remaining_task_count"] < review_stats(config)["task_count"] - 1


def test_generated_qc_tasks_are_low_priority(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_fixture_artifacts(config)
    generate_review_tasks(config)

    task_priorities = [
        int(json.loads(line)["priority"])
        for line in (config.paths.data_dir / "review" / "human_loop_tasks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if '"hidden_qc": true' in line
    ]

    assert task_priorities
    assert max(task_priorities) < 0
