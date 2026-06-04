"""Build grounded table-level search documents from pipeline artifacts."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from statvocab.classification import CSV_BY_CATEGORY
from statvocab.classification_features import read_parquet_rows, write_parquet_rows
from statvocab.config import AppConfig
from statvocab.contracts import VocabularyCategory


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    _raise_csv_field_limit()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _json_list(value: object) -> list[str]:
    if not value:
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _term_categories(config: AppConfig) -> dict[str, VocabularyCategory]:
    categories: dict[str, VocabularyCategory] = {}
    for category, filename in CSV_BY_CATEGORY.items():
        for row in _read_csv(config.paths.outputs_dir / filename):
            term_id = row.get("term_id", "").strip()
            if term_id:
                categories[term_id] = category
    return categories


def _domains_by_measure(config: AppConfig) -> dict[str, str]:
    domains: dict[str, str] = {}
    for row in _read_csv(config.paths.outputs_dir / "measure_clusters.csv"):
        term_id = row.get("term_id", "").strip()
        domain = row.get("domain", "").strip()
        if term_id and domain:
            domains[term_id] = domain
    return domains


def _title_clean_by_table(config: AppConfig) -> dict[str, str]:
    path = config.paths.processed_dir / "title_terms.parquet"
    if not path.exists():
        return {}
    clean: dict[str, str] = {}
    for row in read_parquet_rows(path):
        if bool(row.get("is_full_title")):
            title = str(row.get("title_clean") or "").strip()
            if title:
                clean[str(row["table_id"])] = title
    return clean


def _group_times(config: AppConfig) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path = config.paths.processed_dir / "table_times.parquet"
    if not path.exists():
        return grouped
    for row in read_parquet_rows(path):
        item = {
            "time_id": row["time_id"],
            "raw_value": row["raw_value"],
            "normalized_value": row["normalized_value"],
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "granularity": row["granularity"],
            "source_area": row["source_area"],
            "location": row["location"],
            "confidence": row["confidence"],
        }
        grouped[str(row["table_id"])].append(item)
    return grouped


def _group_geographies(config: AppConfig) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path = config.paths.processed_dir / "table_geographies.parquet"
    if not path.exists():
        return grouped
    rows = read_parquet_rows(path)
    preferred = [row for row in rows if row.get("variant") == "enhanced"]
    selected = preferred or rows
    for row in selected:
        item = {
            "geography_id": row["geography_id"],
            "raw_value": row["raw_value"],
            "normalized_value": row["normalized_value"],
            "matching_key": row["matching_key"],
            "code": row["code"],
            "name": row["name"],
            "level": row["level"],
            "country": row["country"],
            "match_method": row["match_method"],
            "source_area": row["source_area"],
            "location": row["location"],
            "confidence": row["confidence"],
        }
        grouped[str(row["table_id"])].append(item)
    return grouped


def _group_vocabulary(
    config: AppConfig,
    categories: dict[str, VocabularyCategory],
    domains: dict[str, str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "measures": [],
            "dimension_names": [],
            "dimension_values": [],
            "units": [],
            "other_ambiguous": [],
            "domains": set(),
            "evidence": [],
        }
    )
    path = config.paths.processed_dir / "table_vocabulary.parquet"
    if not path.exists():
        return grouped

    key_by_category = {
        VocabularyCategory.MEASURE: "measures",
        VocabularyCategory.DIMENSION_NAME: "dimension_names",
        VocabularyCategory.DIMENSION_VALUE: "dimension_values",
        VocabularyCategory.UNIT: "units",
        VocabularyCategory.OTHER_AMBIGUOUS: "other_ambiguous",
    }
    for row in read_parquet_rows(path):
        table_id = str(row["table_id"])
        term_id = str(row["term_id"])
        category = categories.get(term_id)
        if category is None:
            continue
        term = str(row["canonical_term"])
        key = key_by_category[category]
        if term not in grouped[table_id][key]:
            grouped[table_id][key].append(term)
        occurrence_ids = _json_list(row.get("occurrence_ids_json"))
        grouped[table_id]["evidence"].append(
            {
                "kind": "vocabulary",
                "term_id": term_id,
                "term": term,
                "category": category.value,
                "occurrence_ids": occurrence_ids,
                "source_areas": _json_list(row.get("source_areas_json")),
            }
        )
        if category == VocabularyCategory.MEASURE and term_id in domains:
            grouped[table_id]["domains"].add(domains[term_id])

    for payload in grouped.values():
        for key in (
            "measures",
            "dimension_names",
            "dimension_values",
            "units",
            "other_ambiguous",
        ):
            payload[key] = sorted(set(payload[key]), key=str.casefold)
        payload["domains"] = sorted(payload["domains"], key=str.casefold)
        payload["evidence"] = sorted(
            payload["evidence"],
            key=lambda item: (str(item["category"]), str(item["term"]).casefold()),
        )
    return grouped


def _terms_text(values: list[str]) -> str:
    return " ".join(value for value in values if value).strip()


def build_search_documents(config: AppConfig, *, run_id: str) -> tuple[list[dict[str, Any]], Path]:
    """Create one grounded search document per parsed table."""

    tables_path = config.paths.processed_dir / "tables.parquet"
    tables = read_parquet_rows(tables_path)
    categories = _term_categories(config)
    domains = _domains_by_measure(config)
    vocabulary = _group_vocabulary(config, categories, domains)
    times = _group_times(config)
    geographies = _group_geographies(config)
    clean_titles = _title_clean_by_table(config)

    rows: list[dict[str, Any]] = []
    for table in sorted(tables, key=lambda row: str(row["table_id"])):
        table_id = str(table["table_id"])
        title_raw = str(table.get("title") or "")
        title_clean = clean_titles.get(table_id, title_raw).strip()
        vocab = vocabulary.get(
            table_id,
            {
                "measures": [],
                "dimension_names": [],
                "dimension_values": [],
                "units": [],
                "other_ambiguous": [],
                "domains": [],
                "evidence": [],
            },
        )
        geo_rows = sorted(
            geographies.get(table_id, []),
            key=lambda row: (str(row["code"]), str(row["name"])),
        )
        time_rows = sorted(
            times.get(table_id, []),
            key=lambda row: (str(row["start_date"]), str(row["end_date"]), str(row["raw_value"])),
        )
        geography_text = _terms_text(
            sorted(
                {
                    str(item["code"])
                    for item in geo_rows
                    if item.get("code")
                }
                | {
                    str(item["name"])
                    for item in geo_rows
                    if item.get("name")
                },
                key=str.casefold,
            )
        )
        time_text = _terms_text(
            sorted({str(item["normalized_value"]) for item in time_rows}, key=str.casefold)
        )
        classified_text = _terms_text(
            [
                _terms_text(vocab["measures"]),
                _terms_text(vocab["dimension_names"]),
                _terms_text(vocab["dimension_values"]),
                _terms_text(vocab["units"]),
                _terms_text(vocab["other_ambiguous"]),
            ]
        )
        semantic_text = _terms_text(
            [
                title_clean,
                _terms_text(vocab["measures"]),
                _terms_text(vocab["dimension_names"]),
                _terms_text(vocab["dimension_values"]),
                _terms_text(vocab["units"]),
                _terms_text(vocab["domains"]),
            ]
        )
        all_vocabulary = _terms_text([title_clean, classified_text, geography_text, time_text])
        evidence = [
            {"kind": "title", "table_id": table_id, "raw_value": title_raw, "location": "title"},
            *vocab["evidence"],
            *({"kind": "geography", **item} for item in geo_rows),
            *({"kind": "time", **item} for item in time_rows),
        ]
        rows.append(
            {
                "table_id": table_id,
                "title_raw": title_raw,
                "title_clean": title_clean,
                "source_url": str(table.get("source_url") or ""),
                "parse_status": str(table.get("parse_status") or ""),
                "measures_json": _json(vocab["measures"]),
                "dimension_names_json": _json(vocab["dimension_names"]),
                "dimension_values_json": _json(vocab["dimension_values"]),
                "units_json": _json(vocab["units"]),
                "other_ambiguous_json": _json(vocab["other_ambiguous"]),
                "geographies_json": _json(geo_rows),
                "times_json": _json(time_rows),
                "domains_json": _json(vocab["domains"]),
                "evidence_json": _json(evidence),
                "title_lexical": title_clean,
                "all_vocabulary_lexical": all_vocabulary,
                "semantic_text": semantic_text,
                "run_id": run_id,
            }
        )

    path = config.paths.processed_dir / "search_documents.parquet"
    return rows, write_parquet_rows(rows, path)
