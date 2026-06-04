"""Retrieval evaluation for Milestone 6 search."""

from __future__ import annotations

import csv
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from statvocab.classification_features import write_csv_rows
from statvocab.config import AppConfig
from statvocab.contracts import stable_id
from statvocab.search.engine import SearchEngine, SearchSystem

LABEL_SCORES = {"highly_relevant": 2.0, "relevant": 1.0}
PREDICTION_FIELDS = [
    "system",
    "split",
    "question_set",
    "question_id",
    "rank",
    "table_id",
    "score",
    "target_table_id",
    "relevance_label",
]


@dataclass(frozen=True)
class Question:
    """One STAR retrieval question."""

    question_id: str
    question: str
    target_table_id: str
    question_set: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_questions(config: AppConfig) -> list[Question]:
    star_dir = config.paths.external_dir / "star"
    questions: list[Question] = []
    for filename, question_set in (("S_i.csv", "original"), ("S_r.csv", "reformulated")):
        for row in _read_csv(star_dir / filename):
            questions.append(
                Question(
                    question_id=str(row["question_id"]),
                    question=str(row["question"]),
                    target_table_id=str(row["table_id"]),
                    question_set=question_set,
                )
            )
    return questions


def _load_annotations(config: AppConfig) -> dict[tuple[str, str], str]:
    annotations: dict[tuple[str, str], str] = {}
    for row in _read_csv(config.paths.external_dir / "star" / "annotations.csv"):
        annotations[(str(row["question_id"]), str(row["table_id"]))] = str(row["label"])
    return annotations


def _split_question_ids(config: AppConfig, questions: list[Question]) -> dict[str, str]:
    question_ids = sorted(
        {question.question_id for question in questions},
        key=lambda value: int(value),
    )
    rng = random.Random(config.random_seed)
    rng.shuffle(question_ids)
    total = len(question_ids)
    development_cut = round(total * config.search.development_fraction)
    validation_cut = development_cut + round(total * config.search.validation_fraction)
    split_by_id: dict[str, str] = {}
    for index, question_id in enumerate(question_ids):
        if index < development_cut:
            split = "development"
        elif index < validation_cut:
            split = "validation"
        else:
            split = "final_test"
        split_by_id[question_id] = split
    return split_by_id


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _metrics(
    rows_by_question: dict[str, list[dict[str, Any]]],
    questions: list[Question],
    annotations: dict[tuple[str, str], str],
    latencies: list[float],
) -> dict[str, Any]:
    hit_counts = {1: 0, 5: 0, 10: 0}
    relevance_1: list[float] = []
    relevance_5: list[float] = []
    reciprocal_ranks: list[float] = []
    for question in questions:
        rows = rows_by_question.get(question.question_id, [])
        ranked_ids = [str(row["table_id"]) for row in rows]
        for cutoff in hit_counts:
            if question.target_table_id in ranked_ids[:cutoff]:
                hit_counts[cutoff] += 1
        relevance_scores = [
            LABEL_SCORES.get(annotations.get((question.question_id, table_id), ""), 0.0)
            for table_id in ranked_ids
        ]
        relevance_1.append((relevance_scores[0] / 2.0) if relevance_scores else 0.0)
        relevance_5.append((max(relevance_scores[:5]) / 2.0) if relevance_scores else 0.0)
        first_relevant = next(
            (index for index, score in enumerate(relevance_scores, start=1) if score > 0.0),
            None,
        )
        reciprocal_ranks.append(1.0 / first_relevant if first_relevant else 0.0)
    total = len(questions)
    return {
        "question_count": total,
        "HitRate@1": hit_counts[1] / total if total else 0.0,
        "HitRate@5": hit_counts[5] / total if total else 0.0,
        "HitRate@10": hit_counts[10] / total if total else 0.0,
        "Relevance@1": statistics.fmean(relevance_1) if relevance_1 else 0.0,
        "Relevance@5": statistics.fmean(relevance_5) if relevance_5 else 0.0,
        "MRR": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "p50_latency_ms": (_percentile(latencies, 0.50) or 0.0) * 1000.0,
        "p95_latency_ms": (_percentile(latencies, 0.95) or 0.0) * 1000.0,
    }


def _weight_presets(config: AppConfig) -> dict[str, dict[str, float]]:
    current = config.search
    return {
        "prd_default": {
            "semantic_weight": current.semantic_weight,
            "lexical_weight": current.lexical_weight,
            "measure_weight": current.measure_weight,
            "dimension_weight": current.dimension_weight,
            "geography_weight": current.geography_weight,
            "time_weight": current.time_weight,
        },
        "lexical_heavy": {
            "semantic_weight": 0.20,
            "lexical_weight": 0.45,
            "measure_weight": 0.15,
            "dimension_weight": 0.10,
            "geography_weight": 0.07,
            "time_weight": 0.03,
        },
        "semantic_heavy": {
            "semantic_weight": 0.55,
            "lexical_weight": 0.15,
            "measure_weight": 0.10,
            "dimension_weight": 0.08,
            "geography_weight": 0.08,
            "time_weight": 0.04,
        },
        "structured_heavy": {
            "semantic_weight": 0.25,
            "lexical_weight": 0.20,
            "measure_weight": 0.25,
            "dimension_weight": 0.15,
            "geography_weight": 0.10,
            "time_weight": 0.05,
        },
        "geography_time_heavy": {
            "semantic_weight": 0.25,
            "lexical_weight": 0.20,
            "measure_weight": 0.10,
            "dimension_weight": 0.10,
            "geography_weight": 0.25,
            "time_weight": 0.10,
        },
    }


def _run_questions(
    engine: SearchEngine,
    questions: list[Question],
    *,
    system: SearchSystem,
    split: str,
    annotations: dict[tuple[str, str], str],
    limit: int = 10,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[float]]:
    rows_by_question: dict[str, list[dict[str, Any]]] = {}
    prediction_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for question in questions:
        started = time.perf_counter()
        _parsed, ranked = engine.rank(question.question, limit=limit, system=system)
        latencies.append(time.perf_counter() - started)
        rows: list[dict[str, Any]] = []
        for rank, candidate in enumerate(ranked, start=1):
            label = annotations.get((question.question_id, candidate.table_id), "")
            row = {
                "system": system,
                "split": split,
                "question_set": question.question_set,
                "question_id": question.question_id,
                "rank": rank,
                "table_id": candidate.table_id,
                "score": f"{candidate.score:.6f}",
                "target_table_id": question.target_table_id,
                "relevance_label": label,
            }
            rows.append(row)
            prediction_rows.append(row)
        rows_by_question[question.question_id] = rows
    return rows_by_question, prediction_rows, latencies


def _apply_weights(engine: SearchEngine, weights: dict[str, float]) -> None:
    engine.config = engine.config.model_copy(
        update={"search": engine.config.search.model_copy(update=weights)}
    )


def _latest_index_summary(config: AppConfig) -> dict[str, Any]:
    current_path = config.paths.outputs_dir / "search" / "current_index.json"
    if not current_path.exists():
        return {}
    current = json.loads(current_path.read_text(encoding="utf-8"))
    manifest = json.loads(Path(str(current["index_manifest"])).read_text(encoding="utf-8"))
    summary_path = Path(str(manifest["summary"]))
    if not summary_path.exists():
        return {}
    return cast(dict[str, Any], json.loads(summary_path.read_text(encoding="utf-8")))


def evaluate_retrieval(config: AppConfig) -> tuple[Path, dict[str, Any]]:
    """Evaluate retrieval systems against STAR questions and annotations."""

    questions = _load_questions(config)
    annotations = _load_annotations(config)
    split_by_id = _split_question_ids(config, questions)
    validation_reformulated = [
        question
        for question in questions
        if split_by_id.get(question.question_id) == "validation"
        and question.question_set == "reformulated"
    ]
    engine = SearchEngine(config)
    tuning_results: dict[str, dict[str, Any]] = {}
    for preset_name, weights in _weight_presets(config).items():
        _apply_weights(engine, weights)
        rows_by_question, _rows, latencies = _run_questions(
            engine,
            validation_reformulated,
            system="fused",
            split="validation",
            annotations=annotations,
        )
        tuning_results[preset_name] = _metrics(
            rows_by_question,
            validation_reformulated,
            annotations,
            latencies,
        )
    selected_preset = sorted(
        tuning_results,
        key=lambda name: (
            -float(tuning_results[name]["HitRate@10"]),
            -float(tuning_results[name]["Relevance@5"]),
            float(tuning_results[name]["p95_latency_ms"]),
            name,
        ),
    )[0] if tuning_results else "prd_default"
    _apply_weights(engine, _weight_presets(config)[selected_preset])

    final_questions = [
        question
        for question in questions
        if split_by_id.get(question.question_id) == "final_test"
    ]
    systems: tuple[SearchSystem, ...] = (
        "title_bm25",
        "all_vocabulary_bm25",
        "pearl_semantic",
        "fused",
    )
    metrics: dict[str, dict[str, Any]] = {}
    prediction_rows: list[dict[str, Any]] = []
    for system in systems:
        system_metrics: dict[str, Any] = {}
        for question_set in ("original", "reformulated"):
            subset = [
                question for question in final_questions if question.question_set == question_set
            ]
            rows_by_question, rows, latencies = _run_questions(
                engine,
                subset,
                system=system,
                split="final_test",
                annotations=annotations,
            )
            prediction_rows.extend(rows)
            system_metrics[question_set] = _metrics(
                rows_by_question,
                subset,
                annotations,
                latencies,
            )
        metrics[system] = system_metrics

    run_id = stable_id(
        "run",
        config.config_name,
        config.corpus.name,
        "retrieval",
        datetime.now(UTC).isoformat(),
    )
    output_dir = config.paths.outputs_dir / "retrieval" / run_id
    predictions_path = write_csv_rows(
        prediction_rows,
        output_dir / "retrieval_predictions.csv",
        PREDICTION_FIELDS,
    )
    index_summary = _latest_index_summary(config)
    unique_split_counts = {
        split: sum(1 for question_split in split_by_id.values() if question_split == split)
        for split in ("development", "validation", "final_test")
    }
    row_split_counts = {
        split: sum(
            1
            for question in questions
            if split_by_id.get(question.question_id) == split
        )
        for split in ("development", "validation", "final_test")
    }
    summary = {
        "area": "retrieval",
        "run_id": run_id,
        "config_name": config.config_name,
        "corpus": config.corpus.name,
        "question_count": len(questions),
        "question_row_split_counts": row_split_counts,
        "unique_question_id_split_counts": unique_split_counts,
        "systems": metrics,
        "tuning": {
            "split": "validation",
            "question_set": "reformulated",
            "presets": tuning_results,
            "selected_preset": selected_preset,
            "selected_weights": _weight_presets(config)[selected_preset],
        },
        "index": index_summary,
        "predictions": str(predictions_path),
    }
    summary_path = _write_json(summary, output_dir / "retrieval_summary.json")
    summary["summary"] = str(summary_path)
    metrics_path = _write_json(summary, config.paths.reports_dir / "retrieval_metrics.json")
    return metrics_path, summary
