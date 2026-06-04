"""Milestone 2 extraction pipeline for strings, geography, titles, and vocabulary."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from statvocab.config import AppConfig
from statvocab.contracts import ResourceRecord, stable_id
from statvocab.geo_extract import (
    GeographyMatch,
    build_geography_matcher,
    is_geography_metadata_column,
)
from statvocab.ingest import _read_csv_rows
from statvocab.manifests import complete_manifest, create_manifest, write_manifest
from statvocab.normalize import is_numeric_like, normalize_display, normalize_matching_key
from statvocab.time_extract import TimeOccurrence, extract_header_times, extract_title_times
from statvocab.title_extract import TitleTerm, extract_title_terms

MISSING_MARKERS = {"", ":"}


def _read_tables(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run `statvocab ingest --config <config>` before extraction."
        )
    return cast(list[dict[str, Any]], pq.read_table(path).to_pylist())


def _write_parquet(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)
    return output_path


def _write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _path_from_row(row: dict[str, Any]) -> Path:
    value = str(row["parsed_local_path"])
    return Path(value)


def _list_value(row: dict[str, Any], key: str) -> list[str]:
    values = row.get(key) or []
    return [str(value) for value in values]


def _row_string(value: object) -> str:
    return "" if value is None else str(value)


def _make_string_occurrence(
    *,
    table_id: str,
    raw_term: str,
    source_area: str,
    location: str,
    extraction_rule: str,
    metadata_column: str | None,
    row_index: int | None,
    column_index: int | None,
    artifact_run_id: str,
) -> dict[str, Any] | None:
    normalized = normalize_display(raw_term)
    matching_key = normalize_matching_key(normalized)
    if not matching_key:
        return None
    occurrence_id = stable_id("occ", table_id, source_area, location, matching_key)
    return {
        "occurrence_id": occurrence_id,
        "table_id": table_id,
        "raw_term": raw_term,
        "normalized_term": normalized,
        "matching_key": matching_key,
        "source_area": source_area,
        "metadata_column": metadata_column,
        "row_index": row_index,
        "column_index": column_index,
        "source_span_start": None,
        "source_span_end": None,
        "location": location,
        "extraction_rule": extraction_rule,
        "run_id": artifact_run_id,
    }


def _extract_string_occurrences(
    table_row: dict[str, Any],
    *,
    artifact_run_id: str,
) -> tuple[list[dict[str, Any]], int]:
    table_id = _row_string(table_row["table_id"])
    metadata_columns = _list_value(table_row, "metadata_columns")
    rows = _read_csv_rows(_path_from_row(table_row))
    if not rows:
        return [], 0

    header = rows[0]
    metadata_indexes = [
        index for index, column_name in enumerate(header) if column_name in set(metadata_columns)
    ]
    occurrences: list[dict[str, Any]] = []
    for index in metadata_indexes:
        occurrence = _make_string_occurrence(
            table_id=table_id,
            raw_term=header[index],
            source_area="header_name",
            location=f"header[{index}]",
            extraction_rule="metadata_column_name",
            metadata_column=header[index],
            row_index=None,
            column_index=index,
            artifact_run_id=artifact_run_id,
        )
        if occurrence is not None:
            occurrences.append(occurrence)

    malformed_rows = 0
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            malformed_rows += 1
            continue
        for index in metadata_indexes:
            raw_value = row[index]
            stripped = raw_value.strip()
            if stripped in MISSING_MARKERS or is_numeric_like(stripped):
                continue
            occurrence = _make_string_occurrence(
                table_id=table_id,
                raw_term=raw_value,
                source_area="metadata_value",
                location=f"row[{row_number}].col[{index}]",
                extraction_rule="metadata_non_numeric_value",
                metadata_column=header[index],
                row_index=row_number,
                column_index=index,
                artifact_run_id=artifact_run_id,
            )
            if occurrence is not None:
                occurrences.append(occurrence)
    return occurrences, malformed_rows


def _time_to_row(occurrence: TimeOccurrence, artifact_run_id: str) -> dict[str, Any]:
    return {
        "time_id": occurrence.time_id,
        "table_id": occurrence.table_id,
        "raw_value": occurrence.raw_value,
        "normalized_value": occurrence.normalized_value,
        "start_date": occurrence.start_date,
        "end_date": occurrence.end_date,
        "granularity": occurrence.granularity,
        "source_area": occurrence.source_area,
        "location": occurrence.location,
        "source_span_start": occurrence.source_span_start,
        "source_span_end": occurrence.source_span_end,
        "extractor_rule": occurrence.extractor_rule,
        "confidence": occurrence.confidence,
        "run_id": artifact_run_id,
    }


def _geo_to_row(match: GeographyMatch, artifact_run_id: str) -> dict[str, Any]:
    return {
        "geography_id": match.geography_id,
        "table_id": match.table_id,
        "raw_value": match.raw_value,
        "normalized_value": match.normalized_value,
        "matching_key": match.matching_key,
        "variant": match.variant,
        "code": match.code,
        "name": match.name,
        "level": match.level,
        "country": match.country,
        "match_method": match.match_method,
        "source_area": match.source_area,
        "location": match.location,
        "metadata_column": match.metadata_column,
        "row_index": match.row_index,
        "column_index": match.column_index,
        "source_span_start": match.source_span_start,
        "source_span_end": match.source_span_end,
        "confidence": match.confidence,
        "run_id": artifact_run_id,
    }


def _title_term_to_row(term: TitleTerm, title_clean: str, artifact_run_id: str) -> dict[str, Any]:
    return {
        "occurrence_id": term.occurrence_id,
        "table_id": term.table_id,
        "raw_term": term.raw_term,
        "normalized_term": term.normalized_term,
        "matching_key": term.matching_key,
        "source_area": term.source_area,
        "source_span_start": term.source_span_start,
        "source_span_end": term.source_span_end,
        "extraction_rule": term.extraction_rule,
        "is_full_title": term.is_full_title,
        "title_clean": title_clean,
        "run_id": artifact_run_id,
    }


def _term_occurrence_from_string(row: dict[str, Any], term_id: str) -> dict[str, Any]:
    result = dict(row)
    result["term_id"] = term_id
    return result


def _term_occurrence_from_title(row: dict[str, Any], term_id: str) -> dict[str, Any]:
    return {
        "occurrence_id": row["occurrence_id"],
        "term_id": term_id,
        "table_id": row["table_id"],
        "raw_term": row["raw_term"],
        "normalized_term": row["normalized_term"],
        "matching_key": row["matching_key"],
        "source_area": row["source_area"],
        "metadata_column": None,
        "row_index": None,
        "column_index": None,
        "source_span_start": row["source_span_start"],
        "source_span_end": row["source_span_end"],
        "location": "title",
        "extraction_rule": row["extraction_rule"],
        "run_id": row["run_id"],
    }


def _canonical_term(rows: list[dict[str, Any]]) -> str:
    title_rows = [row for row in rows if str(row["source_area"]).startswith("title")]
    candidates = title_rows or rows
    counts = Counter(str(row["normalized_term"]) for row in candidates)
    max_count = max(counts.values())
    winners = [term for term, count in counts.items() if count == max_count]
    return sorted(winners, key=lambda value: value.casefold())[0]


def _role_summary(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["source_area"]) for row in rows).items()))


def _feature_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tables = {str(row["table_id"]) for row in rows}
    metadata_columns = sorted(
        {str(row["metadata_column"]) for row in rows if row.get("metadata_column")}
    )
    return {
        "table_count": len(tables),
        "occurrence_count": len(rows),
        "metadata_columns": metadata_columns,
        "has_title_evidence": any(str(row["source_area"]).startswith("title") for row in rows),
        "has_header_evidence": any(row["source_area"] == "header_name" for row in rows),
        "has_metadata_value_evidence": any(row["source_area"] == "metadata_value" for row in rows),
    }


def _build_table_strings(
    string_occurrences: list[dict[str, Any]],
    *,
    artifact_run_id: str,
) -> list[dict[str, Any]]:
    by_table_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in string_occurrences:
        by_table_key[(str(row["table_id"]), str(row["matching_key"]))].append(row)

    table_strings: list[dict[str, Any]] = []
    for (table_id, matching_key), rows in sorted(by_table_key.items()):
        normalized_counts = Counter(str(row["normalized_term"]) for row in rows)
        max_count = max(normalized_counts.values())
        normalized_term = sorted(
            term for term, count in normalized_counts.items() if count == max_count
        )[0]
        raw_values = sorted({str(row["raw_term"]) for row in rows})
        locations = sorted({str(row["location"]) for row in rows})
        occurrence_ids = sorted({str(row["occurrence_id"]) for row in rows})
        metadata_columns = sorted(
            {str(row["metadata_column"]) for row in rows if row.get("metadata_column")}
        )
        source_areas = sorted({str(row["source_area"]) for row in rows})
        table_strings.append(
            {
                "string_id": stable_id("string", table_id, matching_key),
                "table_id": table_id,
                "raw_values_json": json.dumps(raw_values, sort_keys=True),
                "normalized_term": normalized_term,
                "matching_key": matching_key,
                "occurrence_count": len(rows),
                "occurrence_ids_json": json.dumps(occurrence_ids, sort_keys=True),
                "source_areas_json": json.dumps(source_areas, sort_keys=True),
                "metadata_columns_json": json.dumps(metadata_columns, sort_keys=True),
                "locations_json": json.dumps(locations, sort_keys=True),
                "run_id": artifact_run_id,
            }
        )
    return table_strings


def _build_vocabulary(
    term_occurrences: list[dict[str, Any]],
    *,
    artifact_run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_table_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in term_occurrences:
        by_key[str(row["matching_key"])].append(row)
        by_table_key[(str(row["table_id"]), str(row["matching_key"]))].append(row)

    vocabulary_rows: list[dict[str, Any]] = []
    for matching_key, rows in sorted(by_key.items()):
        term_id = stable_id("term", matching_key)
        canonical = _canonical_term(rows)
        table_ids = sorted({str(row["table_id"]) for row in rows})
        occurrence_ids = sorted({str(row["occurrence_id"]) for row in rows})
        vocabulary_rows.append(
            {
                "term_id": term_id,
                "canonical_term": canonical,
                "matching_key": matching_key,
                "table_count": len(table_ids),
                "occurrence_count": len(rows),
                "occurrence_ids_json": json.dumps(occurrence_ids, sort_keys=True),
                "role_summary_json": json.dumps(_role_summary(rows), sort_keys=True),
                "feature_summary_json": json.dumps(_feature_summary(rows), sort_keys=True),
                "run_id": artifact_run_id,
            }
        )

    table_vocabulary_rows: list[dict[str, Any]] = []
    for (table_id, matching_key), rows in sorted(by_table_key.items()):
        term_id = stable_id("term", matching_key)
        occurrence_ids = sorted({str(row["occurrence_id"]) for row in rows})
        source_areas = sorted({str(row["source_area"]) for row in rows})
        table_vocabulary_rows.append(
            {
                "table_id": table_id,
                "term_id": term_id,
                "canonical_term": _canonical_term(rows),
                "matching_key": matching_key,
                "occurrence_count": len(rows),
                "occurrence_ids_json": json.dumps(occurrence_ids, sort_keys=True),
                "source_areas_json": json.dumps(source_areas, sort_keys=True),
                "role_summary_json": json.dumps(_role_summary(rows), sort_keys=True),
                "run_id": artifact_run_id,
            }
        )

    return table_vocabulary_rows, vocabulary_rows


def _sort_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: tuple(str(row.get(key) or "") for key in keys))


def run_extraction(
    config: AppConfig,
    *,
    resources: Iterable[ResourceRecord] = (),
) -> tuple[dict[str, Path], Path, dict[str, Any]]:
    """Run Milestone 2 extraction and write all artifacts."""

    manifest = create_manifest(config, "extract")
    artifact_run_id = stable_id("run", manifest.config_id, "extract", config.corpus.name)
    table_rows = _read_tables(config.paths.processed_dir / "tables.parquet")
    allow_fixture_fallback = config.corpus.name == "fixture"
    nuts_matcher = build_geography_matcher(
        config.extraction,
        variant="nuts",
        allow_fixture_fallback=allow_fixture_fallback,
    )
    enhanced_matcher = build_geography_matcher(
        config.extraction,
        variant="enhanced",
        allow_fixture_fallback=allow_fixture_fallback,
    )
    active_matcher = (
        enhanced_matcher if config.extraction.geography_variant == "enhanced" else nuts_matcher
    )

    time_rows: list[dict[str, Any]] = []
    string_rows: list[dict[str, Any]] = []
    geography_rows: list[dict[str, Any]] = []
    title_rows: list[dict[str, Any]] = []
    term_occurrences: list[dict[str, Any]] = []
    malformed_rows_skipped = 0

    for table_row in sorted(table_rows, key=lambda row: str(row["table_id"])):
        table_id = _row_string(table_row["table_id"])
        title = _row_string(table_row.get("title")) or None
        time_columns = _list_value(table_row, "time_columns")
        title_times = extract_title_times(table_id, title, config.extraction)
        header_times = extract_header_times(table_id, time_columns, config.extraction)
        time_rows.extend(_time_to_row(item, artifact_run_id) for item in header_times + title_times)

        table_string_rows, malformed_count = _extract_string_occurrences(
            table_row,
            artifact_run_id=artifact_run_id,
        )
        malformed_rows_skipped += malformed_count
        string_rows.extend(table_string_rows)

        active_geo_keys: set[tuple[str, str, str]] = set()
        for string_row in table_string_rows:
            source_area = str(string_row["source_area"])
            location = str(string_row["location"])
            metadata_column = cast(str | None, string_row.get("metadata_column"))
            row_index = cast(int | None, string_row.get("row_index"))
            column_index = cast(int | None, string_row.get("column_index"))
            if is_geography_metadata_column(metadata_column):
                nuts_match = nuts_matcher.match(
                    str(string_row["raw_term"]),
                    table_id=table_id,
                    source_area=source_area,
                    location=location,
                    metadata_column=metadata_column,
                    row_index=row_index,
                    column_index=column_index,
                )
                enhanced_match = enhanced_matcher.match(
                    str(string_row["raw_term"]),
                    table_id=table_id,
                    source_area=source_area,
                    location=location,
                    metadata_column=metadata_column,
                    row_index=row_index,
                    column_index=column_index,
                )
                for match in (nuts_match, enhanced_match):
                    if match is not None:
                        geography_rows.append(_geo_to_row(match, artifact_run_id))
                active_match = active_matcher.match(
                    str(string_row["raw_term"]),
                    table_id=table_id,
                    source_area=source_area,
                    location=location,
                    metadata_column=metadata_column,
                    row_index=row_index,
                    column_index=column_index,
                )
                if active_match is not None:
                    active_geo_keys.add(
                        (
                            str(string_row["table_id"]),
                            str(string_row["source_area"]),
                            str(string_row["location"]),
                        )
                    )

        title_geo_matches = active_matcher.find_in_title(title or "", table_id=table_id)
        geography_rows.extend(_geo_to_row(match, artifact_run_id) for match in title_geo_matches)
        removal_spans = [
            (item.source_span_start, item.source_span_end)
            for item in title_times
            if item.source_span_start is not None and item.source_span_end is not None
        ]
        removal_spans.extend(
            (match.source_span_start, match.source_span_end)
            for match in title_geo_matches
            if match.source_span_start is not None and match.source_span_end is not None
        )
        title_clean, title_terms = extract_title_terms(table_id, title, removal_spans)
        _ = title_clean
        title_artifact_rows = [
            _title_term_to_row(term, title_clean, artifact_run_id) for term in title_terms
        ]
        title_rows.extend(title_artifact_rows)

        for string_row in table_string_rows:
            occurrence_key = (
                str(string_row["table_id"]),
                str(string_row["source_area"]),
                str(string_row["location"]),
            )
            if occurrence_key in active_geo_keys:
                continue
            term_id = stable_id("term", str(string_row["matching_key"]))
            term_occurrences.append(_term_occurrence_from_string(string_row, term_id))
        for title_row in title_artifact_rows:
            term_id = stable_id("term", str(title_row["matching_key"]))
            term_occurrences.append(_term_occurrence_from_title(title_row, term_id))

    table_string_rows = _build_table_strings(
        string_rows,
        artifact_run_id=artifact_run_id,
    )
    table_vocabulary_rows, vocabulary_rows = _build_vocabulary(
        term_occurrences,
        artifact_run_id=artifact_run_id,
    )

    processed_dir = config.paths.processed_dir
    artifacts = {
        "table_times": _write_parquet(
            _sort_rows(time_rows, ("table_id", "source_area", "location", "normalized_value")),
            processed_dir / "table_times.parquet",
        ),
        "table_strings": _write_parquet(
            _sort_rows(table_string_rows, ("table_id", "matching_key")),
            processed_dir / "table_strings.parquet",
        ),
        "table_geographies": _write_parquet(
            _sort_rows(
                geography_rows,
                ("table_id", "variant", "source_area", "location", "matching_key"),
            ),
            processed_dir / "table_geographies.parquet",
        ),
        "title_terms": _write_parquet(
            _sort_rows(title_rows, ("table_id", "source_area", "matching_key")),
            processed_dir / "title_terms.parquet",
        ),
        "term_occurrences": _write_parquet(
            _sort_rows(term_occurrences, ("term_id", "table_id", "occurrence_id")),
            processed_dir / "term_occurrences.parquet",
        ),
        "table_vocabulary": _write_parquet(
            _sort_rows(table_vocabulary_rows, ("table_id", "matching_key")),
            processed_dir / "table_vocabulary.parquet",
        ),
        "vocabulary": _write_parquet(
            _sort_rows(vocabulary_rows, ("matching_key",)),
            processed_dir / "vocabulary.parquet",
        ),
    }

    diagnostics: dict[str, Any] = {
        "config_name": config.config_name,
        "corpus": config.corpus.name,
        "table_count": len(table_rows),
        "time_occurrence_count": len(time_rows),
        "string_occurrence_count": len(string_rows),
        "table_string_count": len(table_string_rows),
        "geography_match_count": len(geography_rows),
        "title_term_count": len(title_rows),
        "term_occurrence_count": len(term_occurrences),
        "table_vocabulary_count": len(table_vocabulary_rows),
        "vocabulary_count": len(vocabulary_rows),
        "malformed_rows_skipped": malformed_rows_skipped,
        "geography_variant_for_vocabulary": config.extraction.geography_variant,
        "artifact_run_id": artifact_run_id,
    }
    diagnostics_path = _write_json(diagnostics, processed_dir / "extraction_diagnostics.json")
    completed = complete_manifest(
        manifest,
        resources=tuple(record.resource_id for record in resources),
        artifacts=tuple(str(path) for path in (*artifacts.values(), diagnostics_path)),
    )
    manifest_path = write_manifest(
        completed,
        config.paths.outputs_dir / "manifests" / f"{manifest.run_id}_extract.json",
    )
    return artifacts | {"diagnostics": diagnostics_path}, manifest_path, diagnostics
