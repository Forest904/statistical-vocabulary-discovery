"""Context evidence features for semantic measure promotion."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from statvocab.config import AppConfig
from statvocab.contracts import VocabularyCategory
from statvocab.title_extract import STATISTICAL_HEADS

DIMENSION_CONTEXT_HINTS = {
    "age",
    "category",
    "classification",
    "geo",
    "geopolitical",
    "frequency",
    "freq",
    "sex",
    "time",
    "unit",
}
TIME_GEO_VALUE_HINTS = {
    "annual",
    "quarterly",
    "monthly",
    "weekly",
    "daily",
    "female",
    "females",
    "from",
    "geo",
    "male",
    "males",
    "region",
    "total",
    "year",
    "years",
}
ECONOMIC_HEADS = {
    "expenditure",
    "gdp",
    "income",
    "price",
    "prices",
    "production",
    "sales",
    "turnover",
    "value added",
}
COUNT_QUANTITY_HEADS = {
    "area",
    "birth",
    "crime",
    "death",
    "employment",
    "number",
    "population",
    "rate",
    "unemployment",
}


def _json_dict(value: object) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _contains_any(value: str, needles: set[str]) -> bool:
    lower = value.casefold()
    return any(needle in lower for needle in needles)


def _has_age_band(value: str) -> bool:
    lower = value.casefold()
    return bool(re.search(r"\b\d+\s*(?:to|-)\s*\d+\s*years?\b", lower)) or (
        "years or over" in lower
    )


def _occurrence_ids(row: dict[str, Any]) -> set[str]:
    return set(_json_list(row.get("occurrence_ids_json")))


def _read_optional_parquet(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    from statvocab.classification_features import read_parquet_rows

    return read_parquet_rows(path)


def _source_counts(
    row: dict[str, Any],
    occurrences: list[dict[str, Any]],
) -> Counter[str]:
    if occurrences:
        return Counter(str(item.get("source_area") or "") for item in occurrences)
    return Counter(
        {
            str(key): int(value)
            for key, value in _json_dict(row["role_summary_json"]).items()
        }
    )


def _metadata_columns(
    row: dict[str, Any],
    occurrences: list[dict[str, Any]],
) -> list[str]:
    columns = {
        str(item.get("metadata_column") or "")
        for item in occurrences
        if item.get("metadata_column")
    }
    columns.update(_json_list(row.get("metadata_columns_json")))
    return sorted(column for column in columns if column)


def measure_subtype_for_context(context: dict[str, object]) -> str:
    """Return a diagnostic-only measure subtype for reporting."""

    if int(context.get("title_keyphrase_count") or 0) > 0:
        return "title_keyphrase_measure"
    if int(context.get("title_clause_count") or 0) > 0:
        return "title_clause_measure"
    if bool(context.get("economic_measure_head")):
        return "economic_measure"
    if bool(context.get("count_quantity_measure_head")):
        return "count_quantity_measure"
    if bool(context.get("statistical_head_match")):
        return "statistical_head_measure"
    return "weak_semantic_measure"


def _vote_features(
    row: dict[str, Any],
    context: dict[str, object],
    semantic: dict[str, object],
) -> dict[str, object]:
    measure_votes = 0
    negative_votes = 0
    if semantic.get("semantic_best_centroid_class") == VocabularyCategory.MEASURE.value:
        measure_votes += 1
    if semantic.get("semantic_neighbor_best_class") == VocabularyCategory.MEASURE.value:
        measure_votes += 1
    if bool(context["title_statistical_head_evidence"]):
        measure_votes += 1
    if bool(context["title_only_evidence"]) and bool(context["statistical_head_match"]):
        measure_votes += 1
    if bool(context["numeric_unit_context"]) and bool(context["statistical_head_match"]):
        measure_votes += 1

    measure_score = float(semantic.get("semantic_centroid_measure") or 0.0)
    dimension_score = float(semantic.get("semantic_centroid_dimension_value") or 0.0)
    if dimension_score > measure_score:
        negative_votes += 1
    if bool(row.get("has_metadata_value_evidence")) or bool(context["metadata_value_dominant"]):
        negative_votes += 1
    if bool(context["time_geography_value_like"]):
        negative_votes += 1
    if bool(context["dimension_context_evidence"]) or bool(row.get("looks_like_dimension_name")):
        negative_votes += 1

    if measure_votes > negative_votes:
        label = VocabularyCategory.MEASURE.value
    elif negative_votes > measure_votes:
        label = VocabularyCategory.DIMENSION_VALUE.value
    else:
        label = "abstain"
    return {
        "measure_vote_count": measure_votes,
        "negative_vote_count": negative_votes,
        "measure_vote_margin": measure_votes - negative_votes,
        "measure_vote_agreement": measure_votes,
        "weak_supervision_label": label,
    }


def _score_features(context: dict[str, object]) -> dict[str, object]:
    measure_votes = int(context["measure_vote_count"])
    negative_votes = int(context["negative_vote_count"])
    title_ratio = float(context["title_occurrence_ratio"])
    acceptance = min(
        1.0,
        (0.18 * measure_votes)
        + (0.20 if context["title_keyphrase_count"] else 0.0)
        + (0.15 if context["statistical_head_match"] else 0.0)
        + (0.12 if context["title_only_evidence"] else 0.0)
        + (0.10 if context["numeric_unit_context"] else 0.0)
        + min(0.08, title_ratio * 0.08),
    )
    negative = min(
        1.0,
        (0.22 * negative_votes)
        + (0.20 if context["dimension_context_evidence"] else 0.0)
        + (0.18 if context["metadata_value_dominant"] else 0.0)
        + (0.16 if context["time_geography_value_like"] else 0.0),
    )
    return {
        "measure_acceptance_score": round(acceptance, 6),
        "negative_dimension_score": round(negative, 6),
    }


def build_measure_context_features(
    config: AppConfig,
    rows: list[dict[str, Any]],
    semantic_features: dict[str, dict[str, object]] | None = None,
    *,
    term_occurrences: list[dict[str, Any]] | None = None,
    title_terms: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, object]]:
    """Build deterministic context and weak-supervision features per term."""

    actual_semantic = semantic_features or {}
    occurrences = term_occurrences
    if occurrences is None:
        occurrences = _read_optional_parquet(
            config.paths.processed_dir / "term_occurrences.parquet"
        )
    titles = title_terms
    if titles is None:
        titles = _read_optional_parquet(config.paths.processed_dir / "title_terms.parquet")

    occurrences_by_term: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in occurrences:
        occurrences_by_term[str(item.get("term_id") or "")].append(item)
    title_rule_by_occurrence = {
        str(item.get("occurrence_id") or ""): str(item.get("extraction_rule") or "")
        for item in titles
    }

    features: dict[str, dict[str, object]] = {}
    for row in rows:
        term_id = str(row["term_id"])
        canonical = str(row["canonical_term"])
        term_occurrence_rows = occurrences_by_term.get(term_id, [])
        source_counts = _source_counts(row, term_occurrence_rows)
        occurrence_count = int(row.get("occurrence_count") or sum(source_counts.values()))
        title_count = sum(
            count for source, count in source_counts.items() if source.startswith("title")
        )
        header_count = int(source_counts.get("header_name", 0))
        metadata_value_count = int(source_counts.get("metadata_value", 0))
        columns = _metadata_columns(row, term_occurrence_rows)
        title_rules = Counter(
            title_rule_by_occurrence.get(occurrence_id, "")
            for occurrence_id in _occurrence_ids(row)
        )
        title_full_count = int(source_counts.get("title_full", 0))
        title_clause_count = sum(
            count for rule, count in title_rules.items() if "clause" in rule
        )
        title_keyphrase_count = int(title_rules.get("title_keyphrase_statistical_head", 0))
        has_statistical_head = _contains_any(canonical, set(STATISTICAL_HEADS))
        context: dict[str, object] = {
            "term_id": term_id,
            "title_full_count": title_full_count,
            "title_clause_count": title_clause_count,
            "title_keyphrase_count": title_keyphrase_count,
            "title_occurrence_count": title_count,
            "metadata_value_count": metadata_value_count,
            "header_count": header_count,
            "title_occurrence_ratio": _safe_ratio(title_count, occurrence_count),
            "metadata_value_ratio": _safe_ratio(metadata_value_count, occurrence_count),
            "header_ratio": _safe_ratio(header_count, occurrence_count),
            "title_only_evidence": (
                title_count > 0 and metadata_value_count == 0 and header_count == 0
            ),
            "metadata_value_dominant": _safe_ratio(metadata_value_count, occurrence_count) >= 0.50,
            "statistical_head_match": has_statistical_head,
            "title_statistical_head_evidence": title_count > 0 and has_statistical_head,
            "economic_measure_head": _contains_any(canonical, ECONOMIC_HEADS),
            "count_quantity_measure_head": _contains_any(canonical, COUNT_QUANTITY_HEADS),
            "numeric_unit_context": bool(row.get("has_percent"))
            or bool(row.get("has_unit_word"))
            or _contains_any(canonical, {"index", "number", "percentage", "rate"}),
            "dimension_context_evidence": any(
                _contains_any(column, DIMENSION_CONTEXT_HINTS) for column in columns
            ),
            "time_geography_value_like": bool(row.get("has_age_pattern"))
            or _has_age_band(canonical)
            or _contains_any(canonical, TIME_GEO_VALUE_HINTS),
        }
        context.update(_vote_features(row, context, actual_semantic.get(term_id, {})))
        context.update(_score_features(context))
        context["measure_subtype"] = measure_subtype_for_context(context)
        features[term_id] = context
    return features


def weak_supervision_vote_summary(
    context_features: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Summarize weak-supervision votes for run diagnostics."""

    labels = Counter(
        str(row.get("weak_supervision_label") or "abstain")
        for row in context_features.values()
    )
    total = len(context_features)
    avg_measure_votes = (
        sum(int(row.get("measure_vote_count") or 0) for row in context_features.values()) / total
        if total
        else 0.0
    )
    avg_negative_votes = (
        sum(int(row.get("negative_vote_count") or 0) for row in context_features.values()) / total
        if total
        else 0.0
    )
    return {
        "label_counts": dict(sorted(labels.items())),
        "average_measure_vote_count": avg_measure_votes,
        "average_negative_vote_count": avg_negative_votes,
    }
