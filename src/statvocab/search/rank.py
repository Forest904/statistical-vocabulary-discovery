"""Candidate union, score normalization, and weighted search fusion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from statvocab.config import SearchConfig
from statvocab.normalize import normalize_matching_key
from statvocab.search.lexical import LexicalHit
from statvocab.search.query_parser import ParsedQuery
from statvocab.search.semantic import SemanticHit


@dataclass(frozen=True)
class RankedCandidate:
    """One fused table candidate."""

    table_id: str
    score: float
    score_components: dict[str, float]
    document: dict[str, Any]


def _load_list(row: dict[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    if not value:
        return []
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, list) else []


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    minimum = min(values)
    maximum = max(values)
    if maximum <= 0.0:
        return dict.fromkeys(scores, 0.0)
    if maximum == minimum:
        return {key: 1.0 if value > 0.0 else 0.0 for key, value in scores.items()}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}


def _term_component(query_text: str, terms: list[Any]) -> float:
    query_key = normalize_matching_key(query_text)
    if not query_key:
        return 0.0
    query_tokens = set(query_key.split())
    matches = 0
    total = 0
    for value in terms:
        term_key = normalize_matching_key(str(value))
        if not term_key:
            continue
        total += 1
        term_tokens = set(term_key.split())
        if term_key in query_key or query_tokens.intersection(term_tokens):
            matches += 1
    if total == 0:
        return 0.0
    return min(1.0, matches / min(total, 5))


def _geography_component(query: ParsedQuery, document: dict[str, Any]) -> float:
    if not query.geographies:
        return 0.0
    doc_geos = _load_list(document, "geographies_json")
    if not doc_geos:
        return 0.0
    for query_geo in query.geographies:
        query_values = {
            normalize_matching_key(query_geo.code),
            normalize_matching_key(query_geo.name),
            normalize_matching_key(query_geo.normalized_value),
        }
        for doc_geo in doc_geos:
            doc_values = {
                normalize_matching_key(str(doc_geo.get("code", ""))),
                normalize_matching_key(str(doc_geo.get("name", ""))),
                normalize_matching_key(str(doc_geo.get("normalized_value", ""))),
            }
            if query_values.intersection(doc_values):
                return 1.0
            if query_geo.country and query_geo.country == str(doc_geo.get("country", "")):
                return 0.7
    return 0.0


def _parse_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _time_component(query: ParsedQuery, document: dict[str, Any]) -> float:
    if not query.times:
        return 0.0
    doc_times = _load_list(document, "times_json")
    if not doc_times:
        return 0.0
    best = 0.0
    for query_time in query.times:
        query_start = _parse_date(query_time.start_date)
        query_end = _parse_date(query_time.end_date)
        if query_start is None or query_end is None:
            continue
        for doc_time in doc_times:
            doc_start = _parse_date(doc_time.get("start_date"))
            doc_end = _parse_date(doc_time.get("end_date"))
            if doc_start is None or doc_end is None:
                continue
            if doc_start <= query_start and doc_end >= query_end:
                best = max(best, 1.0)
            elif doc_start <= query_end and query_start <= doc_end:
                best = max(best, 0.5)
    return best


def _weights(config: SearchConfig, query: ParsedQuery) -> dict[str, float]:
    weights = {
        "semantic": config.semantic_weight,
        "lexical": config.lexical_weight,
        "measure": config.measure_weight,
        "dimension": config.dimension_weight,
    }
    if query.geographies:
        weights["geography"] = config.geography_weight
    if query.times:
        weights["time"] = config.time_weight
    total = sum(weights.values())
    if total <= 0.0:
        return dict.fromkeys(weights, 0.0)
    return {key: value / total for key, value in weights.items()}


def rank_candidates(
    *,
    query: ParsedQuery,
    documents: dict[str, dict[str, Any]],
    lexical_hits: list[LexicalHit],
    semantic_hits: list[SemanticHit],
    config: SearchConfig,
    limit: int | None = None,
) -> list[RankedCandidate]:
    """Union candidates and apply configured weighted score fusion."""

    lexical_raw = {
        hit.table_id: hit.score
        for hit in lexical_hits
        if hit.table_id in documents
    }
    semantic_raw = {
        hit.table_id: max(0.0, hit.score)
        for hit in semantic_hits
        if hit.table_id in documents
    }
    candidate_ids = sorted(set(lexical_raw) | set(semantic_raw))
    lexical = _normalize_scores(lexical_raw)
    semantic = _normalize_scores(semantic_raw)
    weights = _weights(config, query)
    ranked: list[RankedCandidate] = []
    for table_id in candidate_ids:
        document = documents[table_id]
        dimension_terms = [
            *_load_list(document, "dimension_names_json"),
            *_load_list(document, "dimension_values_json"),
            *_load_list(document, "units_json"),
        ]
        components = {
            "semantic": semantic.get(table_id, 0.0),
            "lexical": lexical.get(table_id, 0.0),
            "measure": _term_component(
                query.semantic_remainder,
                _load_list(document, "measures_json"),
            ),
            "dimension": _term_component(query.semantic_remainder, dimension_terms),
            "geography": _geography_component(query, document),
            "time": _time_component(query, document),
        }
        fused = sum(components[key] * weight for key, weight in weights.items())
        if fused < config.min_fused_score:
            continue
        ranked.append(
            RankedCandidate(
                table_id=table_id,
                score=fused,
                score_components=components,
                document=document,
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.table_id))
    return ranked[: limit or config.default_limit]
