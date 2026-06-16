"""Milestone 5 measure relationship candidate generation."""

from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from statvocab.classification_features import (
    feature_rows,
    read_parquet_rows,
    write_csv_rows,
)
from statvocab.classify_embeddings import generate_term_embeddings
from statvocab.config import AppConfig
from statvocab.contracts import MeasureRelation, RelationType, ResourceRecord, stable_id
from statvocab.manifests import complete_manifest, create_manifest, write_manifest

CROSS_DOMAIN = "cross-domain or other"
RELATION_FIELDNAMES = [
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
]
CANDIDATE_FIELDNAMES = [
    *RELATION_FIELDNAMES,
    "accepted",
    "rejection_reason",
]
ACCEPTED_FIELDNAMES = RELATION_FIELDNAMES
REVIEW_FIELDNAMES = [
    *RELATION_FIELDNAMES,
    "is_valid_relation",
    "correct_relation_type",
    "gold_relation_type",
    "false_positive_type",
    "notes",
]
QUALIFIER_WORDS = {
    "age",
    "annual",
    "area",
    "by",
    "female",
    "index",
    "male",
    "monthly",
    "per",
    "percentage",
    "quarterly",
    "rate",
    "regional",
    "rural",
    "sector",
    "sex",
    "total",
    "urban",
}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class _Measure:
    term_id: str
    term: str
    tokens: frozenset[str]
    occurrence_ids: tuple[str, ...]
    domain: str = ""
    cluster_id: str = ""


@dataclass(frozen=True)
class _Candidate:
    source_term_id: str
    target_term_id: str
    relation_type: RelationType
    confidence: float
    evidence_ids: tuple[str, ...]
    evidence: dict[str, Any]
    methods: tuple[str, ...]


def _read_csv(path: Path, *, required: bool = False) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing {path}; run the previous pipeline stage first.")
        return []
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _json_list(value: object) -> tuple[str, ...]:
    if not value:
        return ()
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        return ()
    return tuple(sorted(str(item) for item in parsed))


def _tokens(term: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", term.casefold())
    return frozenset(word for word in words if word not in STOP_WORDS)


def _normalized(term: str) -> str:
    return " ".join(sorted(_tokens(term)))


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator == 0.0:
        return 0.0
    return _dot(left, right) / denominator


def _measure_rows(config: AppConfig) -> list[dict[str, str]]:
    rows = _read_csv(config.paths.outputs_dir / "measures.csv", required=True)
    rows.sort(key=lambda row: (row["canonical_term"].casefold(), row["term_id"]))
    return rows


def _cluster_rows(config: AppConfig) -> dict[str, dict[str, str]]:
    return {
        row["term_id"]: row
        for row in _read_csv(config.paths.outputs_dir / "measure_clusters.csv")
    }


def _measures(config: AppConfig) -> list[_Measure]:
    clusters = _cluster_rows(config)
    rows: list[_Measure] = []
    for row in _measure_rows(config):
        cluster = clusters.get(row["term_id"], {})
        rows.append(
            _Measure(
                term_id=row["term_id"],
                term=row["canonical_term"],
                tokens=_tokens(row["canonical_term"]),
                occurrence_ids=_json_list(row.get("occurrence_ids_json")),
                domain=cluster.get("domain", ""),
                cluster_id=cluster.get("cluster_id", ""),
            )
        )
    return rows


def _ensure_embeddings(config: AppConfig, measures: list[_Measure]) -> dict[str, list[float]]:
    path = config.paths.processed_dir / "term_embeddings.parquet"
    if not path.exists():
        generate_term_embeddings(config, feature_rows(config))
    rows = read_parquet_rows(path)
    by_term = {
        str(row["term_id"]): [float(value) for value in cast(list[Any], row["embedding"])]
        for row in rows
        if row.get("model") == config.classification.pearl_model
        and row.get("revision") == config.classification.pearl_revision
    }
    missing = sorted(measure.term_id for measure in measures if measure.term_id not in by_term)
    if missing:
        generate_term_embeddings(config, feature_rows(config))
        rows = read_parquet_rows(path)
        by_term = {
            str(row["term_id"]): [float(value) for value in cast(list[Any], row["embedding"])]
            for row in rows
            if row.get("model") == config.classification.pearl_model
            and row.get("revision") == config.classification.pearl_revision
        }
        missing = sorted(measure.term_id for measure in measures if measure.term_id not in by_term)
    if missing:
        raise RuntimeError(f"Missing PEARL embeddings for {len(missing)} measure terms.")
    return by_term


def _evidence_ids(left: _Measure, right: _Measure, method: str) -> tuple[str, ...]:
    ids = tuple(sorted(set(left.occurrence_ids + right.occurrence_ids)))
    if ids:
        return ids
    return (stable_id("evidence", left.term_id, right.term_id, method, digest_size=6),)


def _domain_adjustment(left: _Measure, right: _Measure) -> tuple[float, str]:
    if not left.domain or not right.domain:
        return 0.0, "domain_unavailable"
    if left.domain == right.domain and left.domain != CROSS_DOMAIN:
        return 0.05, "same_domain"
    if left.domain != right.domain and CROSS_DOMAIN not in {left.domain, right.domain}:
        return -0.10, "conflicting_domains"
    return 0.0, "domain_weak_or_cross"


def _candidate(
    source: _Measure,
    target: _Measure,
    relation_type: RelationType,
    confidence: float,
    methods: tuple[str, ...],
    evidence: dict[str, Any],
) -> _Candidate:
    return _Candidate(
        source_term_id=source.term_id,
        target_term_id=target.term_id,
        relation_type=relation_type,
        confidence=max(0.0, min(1.0, confidence)),
        evidence_ids=_evidence_ids(source, target, methods[0]),
        evidence=evidence,
        methods=methods,
    )


def _pair_candidate(
    left: _Measure,
    right: _Measure,
    similarity: float,
    config: AppConfig,
) -> _Candidate | None:
    if left.term_id == right.term_id:
        return None
    common = left.tokens & right.tokens
    union = left.tokens | right.tokens
    jaccard = len(common) / len(union) if union else 0.0
    domain_delta, domain_signal = _domain_adjustment(left, right)
    evidence = {
        "embedding_similarity": round(similarity, 6),
        "token_jaccard": round(jaccard, 6),
        "domain_signal": domain_signal,
        "source_domain": left.domain,
        "target_domain": right.domain,
    }

    if (
        _normalized(left.term) == _normalized(right.term)
        and left.term.casefold() != right.term.casefold()
    ):
        return _candidate(
            left,
            right,
            RelationType.VARIANT_OF,
            0.86 + domain_delta,
            ("title_family", "lexical_variant"),
            evidence,
        )
    if jaccard >= 0.92 and similarity >= config.relations.related_similarity_threshold:
        return _candidate(
            left,
            right,
            RelationType.VARIANT_OF,
            0.80 + domain_delta,
            ("title_family", "embedding_similarity"),
            evidence,
        )

    left_extra = right.tokens - left.tokens
    right_extra = left.tokens - right.tokens
    if (
        left.tokens
        and left.tokens < right.tokens
        and len(left_extra) >= config.relations.containment_min_extra_tokens
    ):
        qualifier_signal = bool(left_extra & QUALIFIER_WORDS)
        return _candidate(
            left,
            right,
            RelationType.BROADER_THAN,
            0.78 + (0.05 if qualifier_signal else 0.0) + domain_delta,
            ("lexical_containment", "qualifier_patterns"),
            {**evidence, "qualifier_tokens": sorted(left_extra)},
        )
    if (
        right.tokens
        and right.tokens < left.tokens
        and len(right_extra) >= config.relations.containment_min_extra_tokens
    ):
        qualifier_signal = bool(right_extra & QUALIFIER_WORDS)
        return _candidate(
            right,
            left,
            RelationType.BROADER_THAN,
            0.78 + (0.05 if qualifier_signal else 0.0) + domain_delta,
            ("lexical_containment", "qualifier_patterns"),
            {**evidence, "qualifier_tokens": sorted(right_extra)},
        )

    if similarity >= config.relations.related_similarity_threshold:
        confidence = 0.50 + ((similarity - config.relations.related_similarity_threshold) * 1.2)
        confidence += 0.08 if similarity >= config.relations.embedding_similarity_threshold else 0.0
        confidence += domain_delta
        if (
            domain_signal == "conflicting_domains"
            and similarity < config.relations.embedding_similarity_threshold
        ):
            return None
        return _candidate(
            left,
            right,
            RelationType.RELATED_TO,
            confidence,
            ("embedding_similarity", "domains"),
            evidence,
        )
    return None


def _valid_candidate(
    candidate: _Candidate,
    valid_measure_ids: set[str],
) -> tuple[bool, str]:
    if candidate.source_term_id == candidate.target_term_id:
        return False, "self_relation"
    if (
        candidate.source_term_id not in valid_measure_ids
        or candidate.target_term_id not in valid_measure_ids
    ):
        return False, "unknown_measure"
    if not candidate.evidence_ids:
        return False, "missing_evidence"
    try:
        MeasureRelation(
            relation_id=stable_id(
                "relation",
                candidate.source_term_id,
                candidate.target_term_id,
                candidate.relation_type.value,
            ),
            source_term_id=candidate.source_term_id,
            target_term_id=candidate.target_term_id,
            relation_type=candidate.relation_type,
            evidence_ids=candidate.evidence_ids,
            confidence=candidate.confidence,
        )
    except ValueError as exc:
        return False, str(exc)
    return True, ""


def _dedupe(
    candidates: list[_Candidate],
    valid_measure_ids: set[str],
) -> tuple[list[_Candidate], dict[int, str]]:
    selected: dict[tuple[str, str], _Candidate] = {}
    rejections: dict[int, str] = {}
    for index, candidate in enumerate(candidates):
        valid, reason = _valid_candidate(candidate, valid_measure_ids)
        if not valid:
            rejections[index] = reason
            continue
        key = (
            min(candidate.source_term_id, candidate.target_term_id),
            max(candidate.source_term_id, candidate.target_term_id),
        )
        current = selected.get(key)
        if current is None or candidate.confidence > current.confidence:
            if current is not None:
                old_index = candidates.index(current)
                rejections[old_index] = "duplicate_lower_confidence"
            selected[key] = candidate
        else:
            rejections[index] = "duplicate_lower_confidence"
    rows = sorted(
        selected.values(),
        key=lambda item: (
            -item.confidence,
            item.relation_type.value,
            item.source_term_id,
            item.target_term_id,
        ),
    )
    return rows, rejections


def _relation_row(
    candidate: _Candidate,
    measures_by_id: dict[str, _Measure],
    run_id: str,
) -> dict[str, Any]:
    relation_id = stable_id(
        "relation",
        candidate.source_term_id,
        candidate.target_term_id,
        candidate.relation_type.value,
    )
    return {
        "relation_id": relation_id,
        "source_term_id": candidate.source_term_id,
        "source_term": measures_by_id[candidate.source_term_id].term,
        "target_term_id": candidate.target_term_id,
        "target_term": measures_by_id[candidate.target_term_id].term,
        "relation_type": candidate.relation_type.value,
        "confidence": f"{candidate.confidence:.6f}",
        "evidence_ids_json": json.dumps(candidate.evidence_ids, sort_keys=True),
        "evidence": json.dumps(candidate.evidence, sort_keys=True),
        "generation_methods_json": json.dumps(candidate.methods, sort_keys=True),
        "run_id": run_id,
    }


def _candidate_pairs(
    measures: list[_Measure],
    vectors_by_term: dict[str, list[float]],
    config: AppConfig,
) -> list[tuple[int, int, float]]:
    if len(measures) <= 1:
        return []
    top_k = min(config.relations.semantic_neighbor_k, len(measures) - 1)
    if len(measures) <= top_k + 1:
        return [
            (
                left_index,
                right_index,
                _cosine(
                    vectors_by_term[measures[left_index].term_id],
                    vectors_by_term[measures[right_index].term_id],
                ),
            )
            for left_index in range(len(measures))
            for right_index in range(left_index + 1, len(measures))
        ]
    lexical_pairs: dict[tuple[int, int], float] = {}
    for left_index in range(len(measures)):
        left = measures[left_index]
        for right_index in range(left_index + 1, len(measures)):
            right = measures[right_index]
            normalized_match = (
                _normalized(left.term) == _normalized(right.term)
                and left.term.casefold() != right.term.casefold()
            )
            containment = (
                bool(left.tokens)
                and bool(right.tokens)
                and (left.tokens < right.tokens or right.tokens < left.tokens)
            )
            if normalized_match or containment:
                lexical_pairs[(left_index, right_index)] = _cosine(
                    vectors_by_term[left.term_id],
                    vectors_by_term[right.term_id],
                )
    try:
        numpy = import_module("numpy")
    except ModuleNotFoundError:
        return [
            (
                left_index,
                right_index,
                _cosine(
                    vectors_by_term[measures[left_index].term_id],
                    vectors_by_term[measures[right_index].term_id],
                ),
            )
            for left_index in range(len(measures))
            for right_index in range(left_index + 1, len(measures))
        ]

    matrix = numpy.array([vectors_by_term[measure.term_id] for measure in measures], dtype=float)
    similarities = matrix @ matrix.T
    pairs: dict[tuple[int, int], float] = {}
    for left_index in range(len(measures)):
        row = similarities[left_index]
        neighbor_indexes = numpy.argpartition(row, -(top_k + 1))[-(top_k + 1) :]
        for right_index_raw in neighbor_indexes:
            right_index = int(right_index_raw)
            if right_index == left_index:
                continue
            key = (min(left_index, right_index), max(left_index, right_index))
            pairs[key] = max(float(row[right_index]), pairs.get(key, -1.0))
    pairs.update(lexical_pairs)
    return [
        (left_index, right_index, similarity)
        for (left_index, right_index), similarity in sorted(pairs.items())
    ]


def _adjudicate_candidates(
    candidates: list[_Candidate],
    config: AppConfig,
) -> list[_Candidate]:
    """Placeholder for optional grounded LLM relation adjudication."""

    _ = config
    return candidates


def _generate_candidates(
    measures: list[_Measure],
    vectors_by_term: dict[str, list[float]],
    config: AppConfig,
) -> list[_Candidate]:
    candidates_by_measure: dict[str, list[_Candidate]] = defaultdict(list)
    for left_index, right_index, similarity in _candidate_pairs(measures, vectors_by_term, config):
        left = measures[left_index]
        right = measures[right_index]
        candidate = _pair_candidate(left, right, similarity, config)
        if candidate is None:
            continue
        candidates_by_measure[candidate.source_term_id].append(candidate)
        candidates_by_measure[candidate.target_term_id].append(candidate)

    selected: dict[tuple[str, str, RelationType], _Candidate] = {}
    for term_candidates in candidates_by_measure.values():
        ranked = sorted(term_candidates, key=lambda candidate: -candidate.confidence)
        for candidate in ranked[: config.relations.max_candidates_per_measure]:
            key = (
                candidate.source_term_id,
                candidate.target_term_id,
                candidate.relation_type,
            )
            current = selected.get(key)
            if current is None or candidate.confidence > current.confidence:
                selected[key] = candidate
    return _adjudicate_candidates(list(selected.values()), config)


def _review_rows(rows: list[dict[str, Any]], config: AppConfig) -> list[dict[str, Any]]:
    sample_size = min(config.relations.manual_review_sample_size, len(rows))
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["relation_type"])].append(row)
    selected: dict[str, dict[str, Any]] = {}
    for relation_type in sorted(by_type):
        ranked = sorted(
            by_type[relation_type],
            key=lambda row: (-float(row["confidence"]), row["relation_id"]),
        )
        for row in ranked[: min(10, len(ranked))]:
            selected[str(row["relation_id"])] = row
    remaining = [row for row in rows if str(row["relation_id"]) not in selected]
    remaining.sort(key=lambda row: str(row["relation_id"]))
    rng = random.Random(config.random_seed)
    if len(selected) < sample_size and remaining:
        for row in rng.sample(remaining, k=min(sample_size - len(selected), len(remaining))):
            selected[str(row["relation_id"])] = row
    review = []
    for row in sorted(
        selected.values(),
        key=lambda item: (-float(item["confidence"]), item["relation_id"]),
    ):
        review.append(
            {
                **row,
                "is_valid_relation": "",
                "correct_relation_type": "",
                "gold_relation_type": "",
                "false_positive_type": "",
                "notes": "",
            }
        )
    return review


def _summary(
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    run_id: str,
    *,
    full_accepted_count: int | None = None,
    llm_adjudication_enabled: bool = False,
) -> dict[str, Any]:
    type_counts = Counter(str(row["relation_type"]) for row in rows)
    method_counts: Counter[str] = Counter()
    confidence_bands: Counter[str] = Counter()
    rejection_counts = Counter(
        str(row["rejection_reason"]) for row in candidate_rows if row.get("rejection_reason")
    )
    for row in rows:
        for method in json.loads(str(row["generation_methods_json"])):
            method_counts[str(method)] += 1
        confidence = float(row["confidence"])
        if confidence >= 0.85:
            confidence_bands["0.85-1.00"] += 1
        elif confidence >= 0.70:
            confidence_bands["0.70-0.84"] += 1
        else:
            confidence_bands["0.00-0.69"] += 1
    return {
        "run_id": run_id,
        "candidate_count": len(candidate_rows),
        "accepted_count": len(rows),
        "full_accepted_count": (
            full_accepted_count if full_accepted_count is not None else len(rows)
        ),
        "relation_type_distribution": dict(sorted(type_counts.items())),
        "generation_method_distribution": dict(sorted(method_counts.items())),
        "rejection_distribution": dict(sorted(rejection_counts.items())),
        "confidence_bands": dict(sorted(confidence_bands.items())),
        "llm_adjudication_enabled": llm_adjudication_enabled,
    }


def run_measure_relations(
    config: AppConfig,
    *,
    resources: tuple[ResourceRecord, ...] = (),
) -> tuple[dict[str, Path], Path, dict[str, Any]]:
    """Generate grounded candidate relationships between final measures."""

    manifest = create_manifest(config, "relations")
    run_id = manifest.run_id
    output_dir = config.paths.outputs_dir / "relations" / run_id
    measures = _measures(config)
    measures_by_id = {measure.term_id: measure for measure in measures}
    valid_measure_ids = set(measures_by_id)

    if measures:
        vectors_by_term = _ensure_embeddings(config, measures)
        candidates = _generate_candidates(measures, vectors_by_term, config)
    else:
        candidates = []

    accepted, rejections = _dedupe(candidates, valid_measure_ids)
    full_relation_rows = [
        _relation_row(candidate, measures_by_id, run_id) for candidate in accepted
    ]
    relation_rows = [
        row
        for row in full_relation_rows
        if row["relation_type"] != RelationType.RELATED_TO.value
        or float(row["confidence"]) >= config.relations.submitted_related_confidence_threshold
    ]
    candidate_rows = []
    for index, candidate in enumerate(candidates):
        row = _relation_row(candidate, measures_by_id, run_id)
        rejection_reason = rejections.get(index, "")
        row["accepted"] = str(not rejection_reason).lower()
        row["rejection_reason"] = rejection_reason
        candidate_rows.append(row)

    review_rows = _review_rows(relation_rows, config)
    summary = _summary(
        relation_rows,
        candidate_rows,
        run_id,
        full_accepted_count=len(full_relation_rows),
        llm_adjudication_enabled=config.relations.llm_adjudication_enabled,
    )
    summary["submitted_related_confidence_threshold"] = (
        config.relations.submitted_related_confidence_threshold
    )
    artifacts = {
        "measure_relations": write_csv_rows(
            relation_rows,
            config.paths.outputs_dir / "measure_relations.csv",
            RELATION_FIELDNAMES,
        ),
        "accepted_relations_full": write_csv_rows(
            full_relation_rows,
            output_dir / "accepted_relations_full.csv",
            ACCEPTED_FIELDNAMES,
        ),
        "relation_candidates": write_csv_rows(
            candidate_rows,
            output_dir / "relation_candidates.csv",
            CANDIDATE_FIELDNAMES,
        ),
        "manual_review_sample": write_csv_rows(
            review_rows,
            output_dir / "manual_relation_review_sample.csv",
            REVIEW_FIELDNAMES,
        ),
        "summary": _write_json(summary, output_dir / "relation_summary.json"),
    }
    completed = complete_manifest(
        manifest,
        resources=tuple(record.resource_id for record in resources),
        artifacts=tuple(str(path) for path in artifacts.values()),
    )
    manifest_path = write_manifest(
        completed,
        config.paths.outputs_dir / "manifests" / f"{manifest.run_id}_relations.json",
    )
    return artifacts, manifest_path, summary
