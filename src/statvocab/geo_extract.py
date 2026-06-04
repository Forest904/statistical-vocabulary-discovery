"""Geography dictionary loading and exact matching."""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from statvocab.config import ExtractionConfig
from statvocab.contracts import stable_id
from statvocab.normalize import normalize_display, normalize_matching_key


@dataclass(frozen=True)
class GeographyEntry:
    """One official geography dictionary entry."""

    code: str
    name: str
    level: str
    country: str
    source: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class GeographyMatch:
    """One accepted geography match against an official dictionary."""

    geography_id: str
    table_id: str
    raw_value: str
    normalized_value: str
    matching_key: str
    variant: str
    code: str
    name: str
    level: str
    country: str
    match_method: str
    source_area: str
    location: str
    metadata_column: str | None
    row_index: int | None
    column_index: int | None
    source_span_start: int | None
    source_span_end: int | None
    confidence: float


FIXTURE_GEO_ENTRIES: tuple[GeographyEntry, ...] = (
    GeographyEntry("DE", "Germany", "country", "DE", "fixture", ("Germany",)),
    GeographyEntry("ES", "Spain", "country", "ES", "fixture", ("Spain",)),
    GeographyEntry("FR", "France", "country", "FR", "fixture", ("France",)),
    GeographyEntry("IT", "Italy", "country", "IT", "fixture", ("Italy",)),
    GeographyEntry("RO", "Romania", "country", "RO", "fixture", ("Romania",)),
    GeographyEntry(
        "EU27_2020",
        "European Union - 27 countries (from 2020)",
        "aggregate",
        "EU",
        "fixture",
        ("EU27_2020",),
    ),
)
COMMON_COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "DE": ("Germany",),
}


def is_geography_metadata_column(column_name: str | None) -> bool:
    """Return whether a metadata column is intended to carry geography values."""

    if not column_name:
        return False
    normalized = normalize_matching_key(column_name)
    return (
        normalized == "geo"
        or normalized.startswith("geo\\")
        or "geopolitical entity" in normalized
    )


def infer_nuts_level(nuts_id: str) -> str:
    """Infer a NUTS level label from a NUTS identifier."""

    length = len(nuts_id.strip())
    if length == 2:
        return "country"
    if length in {3, 4, 5}:
        return f"nuts{length - 2}"
    return "unknown"


def load_nuts_entries(
    path: Path,
    *,
    allow_fixture_fallback: bool = False,
) -> tuple[GeographyEntry, ...]:
    """Load NUTS 2024 entries from the acquired GISCO CSV."""

    if not path.exists():
        return FIXTURE_GEO_ENTRIES if allow_fixture_fallback else ()

    entries: list[GeographyEntry] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            code = (row.get("NUTS_ID") or "").strip()
            country = (row.get("CNTR_CODE") or "").strip()
            name_latn = normalize_display(row.get("NAME_LATN") or "")
            nuts_name = normalize_display(row.get("NUTS_NAME") or "")
            name = name_latn or nuts_name or code
            if not code or not name:
                continue
            aliases = tuple(
                sorted(
                    {
                        alias
                        for alias in (
                            code,
                            name_latn,
                            nuts_name,
                            *COMMON_COUNTRY_ALIASES.get(code, ()),
                        )
                        if alias
                    }
                )
            )
            entries.append(
                GeographyEntry(
                    code=code,
                    name=name,
                    level=infer_nuts_level(code),
                    country=country or code[:2],
                    source="nuts_2024",
                    aliases=aliases,
                )
            )
    return tuple(entries)


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def load_eurostat_geo_entries(path: Path) -> tuple[GeographyEntry, ...]:
    """Load official Eurostat GEO code-list entries from SDMX XML."""

    if not path.exists():
        return ()

    root = ET.parse(path).getroot()
    entries: list[GeographyEntry] = []
    for element in root.iter():
        if _tag_name(element) != "Code":
            continue
        code = (element.attrib.get("id") or element.attrib.get("urn") or "").strip()
        if not code:
            continue
        names = [
            normalize_display(child.text or "")
            for child in element
            if _tag_name(child) == "Name" and normalize_display(child.text or "")
        ]
        name = names[0] if names else code
        aliases = tuple(sorted({code, *names}))
        level = "aggregate" if "_" in code or code in {"EU", "EA"} else infer_nuts_level(code)
        entries.append(
            GeographyEntry(
                code=code,
                name=name,
                level=level,
                country=code[:2] if len(code) >= 2 else "",
                source="eurostat_geo",
                aliases=aliases,
            )
        )
    return tuple(entries)


class GeographyMatcher:
    """Exact raw and normalized matcher for official geography dictionaries."""

    def __init__(self, entries: tuple[GeographyEntry, ...], variant: str) -> None:
        self.variant = variant
        self._raw: dict[str, tuple[GeographyEntry, str]] = {}
        self._normalized: dict[str, tuple[GeographyEntry, str]] = {}
        title_aliases: list[tuple[str, GeographyEntry, str]] = []
        for entry in entries:
            self._add(entry.code, entry, "exact_code")
            for alias in entry.aliases:
                method = "exact_code" if alias == entry.code else "exact_name"
                self._add(alias, entry, method)
                if len(alias) >= 3 or "_" in alias:
                    title_aliases.append((alias, entry, method))
        self._title_aliases = tuple(
            sorted(title_aliases, key=lambda item: (-len(item[0]), item[0].casefold()))
        )

    def _add(self, alias: str, entry: GeographyEntry, method: str) -> None:
        cleaned = normalize_display(alias)
        key = normalize_matching_key(alias)
        if cleaned and cleaned not in self._raw:
            self._raw[cleaned] = (entry, method)
        if key and key not in self._normalized:
            normalized_method = method if method == "exact_code" else "normalized_name"
            self._normalized[key] = (entry, normalized_method)

    def match(
        self,
        raw_value: str,
        *,
        table_id: str,
        source_area: str,
        location: str,
        metadata_column: str | None = None,
        row_index: int | None = None,
        column_index: int | None = None,
        source_span_start: int | None = None,
        source_span_end: int | None = None,
    ) -> GeographyMatch | None:
        """Return an accepted exact geography match, if any."""

        normalized = normalize_display(raw_value)
        key = normalize_matching_key(raw_value)
        found = self._raw.get(normalized)
        if found is None:
            found = self._normalized.get(key)
        if found is None:
            return None

        entry, method = found
        geography_id = stable_id("geo", table_id, self.variant, key, source_area, location)
        return GeographyMatch(
            geography_id=geography_id,
            table_id=table_id,
            raw_value=raw_value,
            normalized_value=normalized,
            matching_key=key,
            variant=self.variant,
            code=entry.code,
            name=entry.name,
            level=entry.level,
            country=entry.country,
            match_method=method,
            source_area=source_area,
            location=location,
            metadata_column=metadata_column,
            row_index=row_index,
            column_index=column_index,
            source_span_start=source_span_start,
            source_span_end=source_span_end,
            confidence=1.0,
        )

    def find_in_title(self, title: str, *, table_id: str) -> list[GeographyMatch]:
        """Find exact official geography names/codes inside a title."""

        matches: list[GeographyMatch] = []
        accepted_spans: list[tuple[int, int]] = []
        tokens = list(re.finditer(r"\b[\w]+(?:[-_][\w]+)*\b", title, re.UNICODE))
        max_ngram = min(8, len(tokens))
        for width in range(max_ngram, 0, -1):
            for start_index in range(0, len(tokens) - width + 1):
                start = tokens[start_index].start()
                end = tokens[start_index + width - 1].end()
                span = (start, end)
                overlaps = any(
                    span[0] < other_end and other_start < span[1]
                    for other_start, other_end in accepted_spans
                )
                if overlaps:
                    continue
                raw_value = title[start:end]
                if len(raw_value) < 3:
                    continue
                key = normalize_matching_key(raw_value)
                found = self._normalized.get(key)
                if found is None:
                    continue
                entry, method = found
                code_like_title_text = (
                    raw_value == entry.code
                    or "_" in raw_value
                    or any(char.isdigit() for char in raw_value)
                )
                if method == "exact_code" and not code_like_title_text:
                    continue
                matches.append(
                    GeographyMatch(
                        geography_id=stable_id(
                            "geo",
                            table_id,
                            self.variant,
                            key,
                            "title",
                            span[0],
                        ),
                        table_id=table_id,
                        raw_value=raw_value,
                        normalized_value=normalize_display(raw_value),
                        matching_key=key,
                        variant=self.variant,
                        code=entry.code,
                        name=entry.name,
                        level=entry.level,
                        country=entry.country,
                        match_method=method if method == "exact_code" else "normalized_name",
                        source_area="title",
                        location="title",
                        metadata_column=None,
                        row_index=None,
                        column_index=None,
                        source_span_start=span[0],
                        source_span_end=span[1],
                        confidence=1.0,
                    )
                )
                accepted_spans.append(span)
        return matches


def build_geography_matcher(
    config: ExtractionConfig,
    *,
    variant: str,
    allow_fixture_fallback: bool = False,
) -> GeographyMatcher:
    """Build the requested geography matcher variant."""

    nuts_entries = load_nuts_entries(
        config.nuts_2024_path,
        allow_fixture_fallback=allow_fixture_fallback,
    )
    entries = list(nuts_entries)
    if variant == "enhanced":
        entries.extend(load_eurostat_geo_entries(config.eurostat_geo_codelist_path))
        if allow_fixture_fallback:
            entries.extend(FIXTURE_GEO_ENTRIES)
    return GeographyMatcher(tuple(entries), variant)
