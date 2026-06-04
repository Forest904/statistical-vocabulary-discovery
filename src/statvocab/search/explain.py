"""Evidence-based search result explanations."""

from __future__ import annotations

import json
from typing import Any

from statvocab.normalize import normalize_matching_key
from statvocab.search.query_parser import ParsedQuery
from statvocab.search.rank import RankedCandidate

NOTICE = "StatVocab finds relevant source tables; it does not return a numeric answer."


def _load_list(row: dict[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    if not value:
        return []
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, list) else []


def _matched_terms(query_text: str, document: dict[str, Any]) -> list[dict[str, str]]:
    query_key = normalize_matching_key(query_text)
    query_tokens = set(query_key.split())
    matches: list[dict[str, str]] = []
    fields = (
        ("measure", "measures_json"),
        ("dimension_name", "dimension_names_json"),
        ("dimension_value", "dimension_values_json"),
        ("unit", "units_json"),
    )
    for category, field in fields:
        for term in _load_list(document, field):
            term_key = normalize_matching_key(str(term))
            if not term_key:
                continue
            if term_key in query_key or query_tokens.intersection(term_key.split()):
                matches.append({"category": category, "term": str(term)})
    return sorted(matches, key=lambda item: (item["category"], item["term"].casefold()))


def _warnings(query: ParsedQuery, candidate: RankedCandidate) -> list[str]:
    warnings: list[str] = []
    components = candidate.score_components
    if query.geographies and components.get("geography", 0.0) == 0.0:
        warnings.append("recognized geography did not match this table")
    if query.geographies and 0.0 < components.get("geography", 0.0) < 1.0:
        warnings.append("recognized geography matched only through a relaxed constraint")
    if query.times and components.get("time", 0.0) == 0.0:
        warnings.append("recognized time did not match this table")
    if query.times and 0.0 < components.get("time", 0.0) < 1.0:
        warnings.append("recognized time matched only through an overlapping interval")
    if not query.geographies:
        warnings.append("no geography constraint recognized in the query")
    if not query.times:
        warnings.append("no time constraint recognized in the query")
    return warnings


def explain_candidate(
    *,
    rank: int,
    query: ParsedQuery,
    candidate: RankedCandidate,
) -> dict[str, Any]:
    """Build a JSON-serializable result with grounded evidence."""

    document = candidate.document
    geographies = _load_list(document, "geographies_json")
    times = _load_list(document, "times_json")
    matched_terms = _matched_terms(query.semantic_remainder, document)
    return {
        "rank": rank,
        "table_id": candidate.table_id,
        "title": document.get("title_clean") or document.get("title_raw") or "",
        "score": round(candidate.score, 6),
        "score_components": {
            key: round(value, 6)
            for key, value in sorted(candidate.score_components.items())
        },
        "matched_terms": matched_terms,
        "geographies": geographies,
        "times": times,
        "source_url": document.get("source_url") or "",
        "evidence": {
            "title": document.get("title_clean") or document.get("title_raw") or "",
            "matched_terms": matched_terms,
            "score_components": candidate.score_components,
            "warnings": _warnings(query, candidate),
        },
        "notice": NOTICE,
    }
