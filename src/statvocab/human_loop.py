"""Human-in-the-loop review tasks and label compilation."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from statvocab.classification import CSV_BY_CATEGORY
from statvocab.classification_evaluate import evaluate_classification
from statvocab.classification_features import (
    CATEGORY_VALUES,
    GOLD_FIELDNAMES,
    read_csv_rows,
    read_parquet_rows,
    write_csv_rows,
)
from statvocab.config import AppConfig

ReviewTaskType = Literal[
    "term_classification",
    "measure_vs_breakdown",
    "title_reclaim",
    "evidence_validation",
    "same_as",
]
ReviewMode = ReviewTaskType | Literal["all"]

CLASSIFICATION_TASK_TYPES = {
    "term_classification",
    "measure_vs_breakdown",
    "title_reclaim",
}
EVENT_FIELDNAMES = [
    "event_id",
    "task_id",
    "task_type",
    "reviewer_id",
    "answer",
    "created_at",
    "term_id",
    "paired_term_id",
    "notes",
]
HUMAN_CLASSIFICATION_FIELDNAMES = [
    *GOLD_FIELDNAMES,
    "audit_source",
    "target_cohort",
    "task_type",
    "task_id",
]
HUMAN_RELATION_FIELDNAMES = [
    "task_id",
    "source_term_id",
    "source_term",
    "target_term_id",
    "target_term",
    "relation_label",
    "relation_type",
    "split",
    "reviewer_id",
    "notes",
]
TASK_CHOICES: dict[str, list[str]] = {
    "term_classification": [
        "measure",
        "dimension_name",
        "dimension_value",
        "unit",
        "other_ambiguous",
        "unsure",
        "skip",
    ],
    "measure_vs_breakdown": ["measure", "dimension_value", "unsure", "skip"],
    "title_reclaim": ["measure", "other_ambiguous", "unsure", "skip"],
    "evidence_validation": ["supports", "rejects", "unsure", "skip"],
    "same_as": ["same", "related", "different", "unsure", "skip"],
}


def review_dir(config: AppConfig) -> Path:
    return config.paths.data_dir / "review"


def tasks_path(config: AppConfig) -> Path:
    return review_dir(config) / "human_loop_tasks.jsonl"


def events_path(config: AppConfig) -> Path:
    return review_dir(config) / "human_loop_events.jsonl"


def _human_classification_path(config: AppConfig) -> Path:
    return config.evaluation.extraction_gold_dir / "human_loop_classification_labels.csv"


def _human_relation_path(config: AppConfig) -> Path:
    return config.evaluation.extraction_gold_dir / "human_loop_relation_labels.csv"


def _human_metrics_path(config: AppConfig) -> Path:
    return config.paths.reports_dir / "human_loop_metrics.json"


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _append_jsonl(row: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.blake2b(
        "\x1f".join(str(part) for part in parts).encode("utf-8"),
        digest_size=10,
    ).hexdigest()
    return f"{prefix}_{digest}"


def _split_for_key(key: str) -> str:
    bucket = int(hashlib.blake2b(key.encode("utf-8"), digest_size=2).hexdigest(), 16) % 100
    if bucket < 70:
        return "train_dev"
    if bucket < 85:
        return "validation"
    return "final_test"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_list(value: object) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _limited_json_list(value: object, *, limit: int = 10) -> list[Any]:
    return _json_list(value)[:limit]


def _safe_read_parquet(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_parquet_rows(path)


def _load_term_outputs(config: AppConfig) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for category, filename in CSV_BY_CATEGORY.items():
        for row in read_csv_rows(config.paths.outputs_dir / filename):
            outputs[str(row["term_id"])] = {**row, "category": category.value}
    return outputs


def _load_features(config: AppConfig) -> dict[str, dict[str, Any]]:
    path = config.paths.processed_dir / "classification_features.parquet"
    if not path.exists():
        return {}
    return {str(row["term_id"]): row for row in read_parquet_rows(path)}


def _table_titles(config: AppConfig) -> dict[str, str]:
    return {
        str(row["table_id"]): str(row.get("title") or row.get("title_clean") or "")
        for row in _safe_read_parquet(config.paths.processed_dir / "tables.parquet")
    }


def _occurrence_context(config: AppConfig) -> dict[str, dict[str, Any]]:
    titles = _table_titles(config)
    context: dict[str, dict[str, Any]] = {}
    for row in _safe_read_parquet(config.paths.processed_dir / "term_occurrences.parquet"):
        term_id = str(row.get("term_id") or "")
        if not term_id or term_id in context:
            continue
        table_id = str(row.get("table_id") or "")
        context[term_id] = {
            "table_id": table_id,
            "table_title": titles.get(table_id, ""),
            "occurrence_id": str(row.get("occurrence_id") or ""),
            "source_area": str(row.get("source_area") or ""),
            "metadata_column": row.get("metadata_column"),
            "location": row.get("location"),
        }
    return context


def _choice_payload(values: list[str]) -> list[dict[str, str]]:
    return [
        {"value": value, "label": value.replace("_", " ").title(), "shortcut": str(index + 1)}
        for index, value in enumerate(values)
    ]


def _task(
    *,
    task_type: str,
    question: str,
    priority: int,
    term: dict[str, Any],
    output: dict[str, Any],
    context: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    paired_term: dict[str, Any] | None = None,
    hidden_qc: bool = False,
) -> dict[str, Any]:
    paired_id = str(paired_term.get("term_id") or "") if paired_term else ""
    task_id = _stable_id(
        "review",
        task_type,
        term["term_id"],
        paired_id,
        "qc" if hidden_qc else "main",
    )
    return {
        "task_id": task_id,
        "task_type": task_type,
        "question": question,
        "term_id": str(term["term_id"]),
        "canonical_term": str(term.get("canonical_term") or output.get("canonical_term") or ""),
        "paired_term_id": paired_id,
        "paired_canonical_term": str(paired_term.get("canonical_term") or "")
        if paired_term
        else "",
        "choices": _choice_payload(TASK_CHOICES[task_type]),
        "context": {
            "current_category": output.get("category"),
            "confidence": output.get("confidence"),
            "evidence": output.get("evidence"),
            "occurrence_ids": _limited_json_list(output.get("occurrence_ids_json"))
            or _limited_json_list(term.get("occurrence_ids_json")),
            **context,
        },
        "metadata": metadata or {},
        "priority": priority,
        "hidden_qc": hidden_qc,
        "created_at": _now_iso(),
    }


def _dedupe_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for task in tasks:
        by_id[str(task["task_id"])] = task
    return sorted(
        by_id.values(),
        key=lambda row: (
            -int(row.get("priority") or 0),
            str(row["task_type"]),
            str(row["canonical_term"]).casefold(),
            str(row["task_id"]),
        ),
    )


def generate_review_tasks(config: AppConfig, *, limit: int = 2000) -> tuple[Path, dict[str, Any]]:
    """Generate a deterministic human-loop task queue from current artifacts."""

    vocabulary = {
        str(row["term_id"]): row
        for row in _safe_read_parquet(config.paths.processed_dir / "vocabulary.parquet")
    }
    outputs = _load_term_outputs(config)
    features = _load_features(config)
    contexts = _occurrence_context(config)
    tasks: list[dict[str, Any]] = []

    for term_id, output in outputs.items():
        term = vocabulary.get(term_id, output)
        feature = features.get(term_id, {})
        context = contexts.get(term_id, {})
        category = str(output.get("category") or "")
        confidence = float(output.get("confidence") or 0.0)
        source_roles = str(feature.get("source_roles") or "")
        has_title = bool(feature.get("has_title_evidence"))
        has_value = bool(feature.get("has_metadata_value_evidence"))
        has_measure_cue = bool(feature.get("has_unit_word")) or has_title
        priority_base = int(term.get("occurrence_count") or 0)

        if category == "other_ambiguous" or confidence < 0.75:
            tasks.append(
                _task(
                    task_type="term_classification",
                    question="What is this term?",
                    priority=1000 + priority_base,
                    term=term,
                    output=output,
                    context=context,
                    metadata={"reason": "other_ambiguous_or_low_confidence"},
                )
            )
        if category in {"other_ambiguous", "dimension_value"} and (has_value or has_measure_cue):
            tasks.append(
                _task(
                    task_type="measure_vs_breakdown",
                    question="Is this a statistic being measured, or a breakdown/category?",
                    priority=900 + priority_base,
                    term=term,
                    output=output,
                    context=context,
                    metadata={"source_roles": source_roles},
                )
            )
        if has_title or "title" in source_roles:
            tasks.append(
                _task(
                    task_type="title_reclaim",
                    question="Does this title phrase name a measure?",
                    priority=800 + priority_base,
                    term=term,
                    output=output,
                    context=context,
                    metadata={"source_roles": source_roles},
                )
            )
        if category != "other_ambiguous" and output.get("evidence"):
            tasks.append(
                _task(
                    task_type="evidence_validation",
                    question="Does this evidence support the model's classification?",
                    priority=300 + priority_base,
                    term=term,
                    output=output,
                    context=context,
                    metadata={"model_category": category},
                )
            )

    relation_rows = read_csv_rows(config.paths.outputs_dir / "measure_relations.csv")
    for row in relation_rows[:200]:
        source_id = str(row.get("source_term_id") or "")
        target_id = str(row.get("target_term_id") or "")
        if not source_id or not target_id:
            continue
        source_term = vocabulary.get(
            source_id,
            {
                "term_id": source_id,
                "canonical_term": row.get("source_term"),
                "occurrence_ids_json": row.get("evidence_ids_json"),
            },
        )
        target_term = vocabulary.get(
            target_id,
            {
                "term_id": target_id,
                "canonical_term": row.get("target_term"),
                "occurrence_ids_json": row.get("evidence_ids_json"),
            },
        )
        tasks.append(
            _task(
                task_type="same_as",
                question="Are these terms the same statistical concept?",
                priority=1200,
                term=source_term,
                output=outputs.get(source_id, {}),
                context=contexts.get(source_id, {}),
                paired_term=target_term,
                metadata={
                    "relation_type": row.get("relation_type"),
                    "confidence": row.get("confidence"),
                },
            )
        )

    previous_tasks = {str(task["task_id"]): task for task in _read_jsonl(tasks_path(config))}
    answered_task_ids = _answered_task_ids(read_review_events(config))
    answered_previous_tasks = [
        previous_tasks[task_id] for task_id in answered_task_ids if task_id in previous_tasks
    ]
    unique = _dedupe_tasks(tasks)[:limit]
    qc_count = min(20, max(1, len(unique) // 100)) if unique else 0
    qc_tasks = [
        {**task, "task_id": f"{task['task_id']}_qc", "hidden_qc": True, "priority": -10}
        for task in unique[:qc_count]
    ]
    final_tasks = _dedupe_tasks([*unique, *qc_tasks, *answered_previous_tasks])
    path = _write_jsonl(final_tasks, tasks_path(config))
    events_path(config).parent.mkdir(parents=True, exist_ok=True)
    events_path(config).touch(exist_ok=True)
    return path, review_stats(config)


def _load_or_generate_tasks(config: AppConfig) -> list[dict[str, Any]]:
    path = tasks_path(config)
    if not path.exists():
        generate_review_tasks(config)
    return _read_jsonl(path)


def read_review_events(config: AppConfig) -> list[dict[str, Any]]:
    return _read_jsonl(events_path(config))


def _answered_task_ids(events: list[dict[str, Any]]) -> set[str]:
    return {str(event["task_id"]) for event in events if event.get("answer")}


def _answered_classification_term_ids(events: list[dict[str, Any]]) -> set[str]:
    return {
        str(event["term_id"])
        for event in events
        if event.get("answer")
        and event.get("task_type") in CLASSIFICATION_TASK_TYPES
        and event.get("term_id")
    }


def _is_task_available(
    task: dict[str, Any],
    *,
    mode: str,
    answered_task_ids: set[str],
    answered_classification_term_ids: set[str],
) -> bool:
    if mode != "all" and task.get("task_type") != mode:
        return False
    if str(task["task_id"]) in answered_task_ids:
        return False
    return not (
        task.get("task_type") in CLASSIFICATION_TASK_TYPES
        and not task.get("hidden_qc")
        and str(task.get("term_id") or "") in answered_classification_term_ids
    )


def next_review_task(config: AppConfig, *, mode: str = "all") -> dict[str, Any] | None:
    """Return the next unanswered task for the requested mode."""

    tasks = _load_or_generate_tasks(config)
    events = read_review_events(config)
    answered = _answered_task_ids(events)
    answered_terms = _answered_classification_term_ids(events)
    for task in tasks:
        if _is_task_available(
            task,
            mode=mode,
            answered_task_ids=answered,
            answered_classification_term_ids=answered_terms,
        ):
            return task
    return None


def append_review_answer(
    config: AppConfig,
    *,
    task_id: str,
    answer: str,
    reviewer_id: str = "local_user",
    notes: str = "",
) -> dict[str, Any]:
    """Append one human review answer event."""

    tasks = {str(task["task_id"]): task for task in _load_or_generate_tasks(config)}
    task = tasks.get(task_id)
    if task is None:
        raise ValueError(f"Unknown review task: {task_id}")
    choices = set(TASK_CHOICES[str(task["task_type"])])
    if answer not in choices:
        raise ValueError(f"Invalid answer {answer!r} for task type {task['task_type']}")
    event = {
        "event_id": _stable_id("event", task_id, reviewer_id, answer, _now_iso()),
        "task_id": task_id,
        "task_type": task["task_type"],
        "reviewer_id": reviewer_id,
        "answer": answer,
        "created_at": _now_iso(),
        "term_id": task.get("term_id") or "",
        "paired_term_id": task.get("paired_term_id") or "",
        "notes": notes,
    }
    _append_jsonl(event, events_path(config))
    return {"event": event, "stats": review_stats(config)}


def _latest_valid_events(config: AppConfig) -> list[dict[str, Any]]:
    events_by_task: dict[str, dict[str, Any]] = {}
    for sequence, event in enumerate(read_review_events(config)):
        answer = str(event.get("answer") or "")
        if answer in {"skip", "unsure", ""}:
            continue
        events_by_task[str(event["task_id"])] = {**event, "_event_sequence": sequence}
    return list(events_by_task.values())


def _existing_category_term_ids(config: AppConfig) -> set[str]:
    existing: set[str] = set()
    for filename in ("vocabulary_gold_labels.csv", "vocabulary_reclaim_labels.csv"):
        for row in read_csv_rows(config.evaluation.extraction_gold_dir / filename):
            if row.get("category") in CATEGORY_VALUES:
                existing.add(str(row["term_id"]))
    return existing


def _locked_splits(config: AppConfig) -> dict[str, str]:
    return {
        str(row["term_id"]): str(row.get("split") or _split_for_key(str(row["term_id"])))
        for row in read_csv_rows(_human_classification_path(config))
        if row.get("term_id")
    }


def compile_human_labels(config: AppConfig) -> tuple[Path, dict[str, Any]]:
    """Compile human-loop events into labels and re-run classification evaluation."""

    tasks = {str(task["task_id"]): task for task in _load_or_generate_tasks(config)}
    valid_events = _latest_valid_events(config)
    existing_term_ids = _existing_category_term_ids(config)
    locked = _locked_splits(config)
    classification_by_term: dict[str, dict[str, Any]] = {}
    relation_rows: list[dict[str, Any]] = []

    for event in sorted(
        valid_events,
        key=lambda row: (int(row.get("_event_sequence") or 0), str(row["task_id"])),
    ):
        task = tasks.get(str(event["task_id"]))
        if task is None:
            continue
        task_type = str(task["task_type"])
        answer = str(event["answer"])
        term_id = str(task.get("term_id") or "")
        if task_type in CLASSIFICATION_TASK_TYPES:
            if term_id in existing_term_ids:
                continue
            split = locked.get(term_id) or _split_for_key(term_id)
            classification_by_term[term_id] = {
                "term_id": term_id,
                "canonical_term": task.get("canonical_term") or "",
                "matching_key": "",
                "split": split,
                "stratum": "human_loop",
                "table_count": "",
                "occurrence_count": "",
                "role_summary_json": "{}",
                "feature_summary_json": "{}",
                "category": answer,
                "annotator_id": event.get("reviewer_id") or "local_user",
                "notes": event.get("notes") or "",
                "audit_source": "human_loop",
                "target_cohort": task_type,
                "task_type": task_type,
                "task_id": task["task_id"],
            }
        elif task_type == "same_as":
            relation_label = {
                "same": "variant_of",
                "related": "related_to",
                "different": "negative",
            }.get(answer)
            if not relation_label:
                continue
            relation_rows.append(
                {
                    "task_id": task["task_id"],
                    "source_term_id": term_id,
                    "source_term": task.get("canonical_term") or "",
                    "target_term_id": task.get("paired_term_id") or "",
                    "target_term": task.get("paired_canonical_term") or "",
                    "relation_label": answer,
                    "relation_type": relation_label,
                    "split": _split_for_key(str(task["task_id"])),
                    "reviewer_id": event.get("reviewer_id") or "local_user",
                    "notes": event.get("notes") or "",
                }
            )

    classification_rows = sorted(
        classification_by_term.values(),
        key=lambda row: (str(row["split"]), str(row["term_id"])),
    )
    relation_rows.sort(key=lambda row: (str(row["split"]), str(row["task_id"])))
    classification_path = write_csv_rows(
        classification_rows,
        _human_classification_path(config),
        HUMAN_CLASSIFICATION_FIELDNAMES,
    )
    relation_path = write_csv_rows(
        relation_rows,
        _human_relation_path(config),
        HUMAN_RELATION_FIELDNAMES,
    )
    metrics_path, classification_metrics = evaluate_classification(config)
    payload = {
        "compiled_at": _now_iso(),
        "events_count": len(read_review_events(config)),
        "compiled_classification_label_count": len(classification_rows),
        "compiled_relation_label_count": len(relation_rows),
        "classification_label_splits": dict(
            Counter(str(row["split"]) for row in classification_rows)
        ),
        "relation_label_splits": dict(Counter(str(row["split"]) for row in relation_rows)),
        "agreement": _agreement_summary(config),
        "outputs": {
            "classification_labels": str(classification_path),
            "relation_labels": str(relation_path),
            "classification_metrics": str(metrics_path),
        },
        "classification_metrics_summary": {
            "gold_status": classification_metrics.get("gold_status"),
            "metrics_status": classification_metrics.get("metrics_status", "available"),
            "macro_f1": (classification_metrics.get("metrics") or {}).get("macro_f1"),
            "promotion_gate_summary": classification_metrics.get("promotion_gate_summary"),
        },
    }
    _write_json(payload, _human_metrics_path(config))
    return _human_metrics_path(config), payload


def _agreement_summary(config: AppConfig) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[str]] = {}
    for event in read_review_events(config):
        if event.get("answer") in {"skip", "unsure", ""}:
            continue
        key = (str(event.get("task_type") or ""), str(event.get("term_id") or ""))
        by_key.setdefault(key, []).append(str(event["answer"]))
    repeated = [answers for answers in by_key.values() if len(answers) > 1]
    disagreements = sum(1 for answers in repeated if len(set(answers)) > 1)
    return {
        "repeated_item_count": len(repeated),
        "disagreement_count": disagreements,
        "agreement_rate": 1.0 - (disagreements / len(repeated)) if repeated else None,
    }


def review_stats(config: AppConfig) -> dict[str, Any]:
    """Return review task/event progress counters."""

    tasks = _read_jsonl(tasks_path(config))
    events = read_review_events(config)
    answered = _answered_task_ids(events)
    answered_terms = _answered_classification_term_ids(events)
    available = [
        task
        for task in tasks
        if _is_task_available(
            task,
            mode="all",
            answered_task_ids=answered,
            answered_classification_term_ids=answered_terms,
        )
    ]
    return {
        "task_count": len(tasks),
        "answered_task_count": len(answered),
        "remaining_task_count": len(available),
        "event_count": len(events),
        "task_type_counts": dict(Counter(str(task.get("task_type") or "") for task in tasks)),
        "answer_counts": dict(Counter(str(event.get("answer") or "") for event in events)),
        "reviewer_counts": dict(Counter(str(event.get("reviewer_id") or "") for event in events)),
        "tasks_path": str(tasks_path(config)),
        "events_path": str(events_path(config)),
    }
