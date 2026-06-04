"""Query parsing for grounded table retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from statvocab.config import AppConfig
from statvocab.geo_extract import GeographyMatch, build_geography_matcher
from statvocab.normalize import collapse_removed_spans, normalize_display
from statvocab.time_extract import TimeOccurrence, extract_title_times


@dataclass(frozen=True)
class ParsedQuery:
    """Structured query representation used by retrieval."""

    original: str
    semantic_remainder: str
    geographies: tuple[GeographyMatch, ...]
    times: tuple[TimeOccurrence, ...]

    @property
    def is_blank(self) -> bool:
        return not self.original.strip()


def _span(value: object) -> tuple[int, int] | None:
    start = getattr(value, "source_span_start", None)
    end = getattr(value, "source_span_end", None)
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return start, end
    return None


def _accepted_non_overlapping(items: list[object]) -> list[object]:
    accepted: list[object] = []
    spans: list[tuple[int, int]] = []
    for item in sorted(
        items,
        key=lambda value: (
            -((_span(value) or (0, 0))[1] - (_span(value) or (0, 0))[0]),
            (_span(value) or (0, 0))[0],
        ),
    ):
        span = _span(item)
        if span is None:
            continue
        if any(span[0] < other_end and other_start < span[1] for other_start, other_end in spans):
            continue
        accepted.append(item)
        spans.append(span)
    return sorted(accepted, key=lambda value: (_span(value) or (0, 0))[0])


def _remove_spans(value: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return normalize_display(value)
    chars = list(value)
    for start, end in spans:
        for index in range(start, end):
            if 0 <= index < len(chars):
                chars[index] = " "
    return collapse_removed_spans("".join(chars))


def parse_query(config: AppConfig, query: str) -> ParsedQuery:
    """Extract time/geography constraints and preserve semantic remainder."""

    original = normalize_display(query)
    if not original:
        return ParsedQuery(original="", semantic_remainder="", geographies=(), times=())

    times = extract_title_times("query", original, config.extraction)
    matcher = build_geography_matcher(
        config.extraction,
        variant="enhanced",
        allow_fixture_fallback=config.corpus.name == "fixture",
    )
    geographies = matcher.find_in_title(original, table_id="query")
    accepted = _accepted_non_overlapping([*times, *geographies])
    accepted_spans = [_span(item) for item in accepted]
    spans = [span for span in accepted_spans if span is not None]
    accepted_time_ids = {id(item) for item in accepted if isinstance(item, TimeOccurrence)}
    accepted_geo_ids = {id(item) for item in accepted if isinstance(item, GeographyMatch)}
    return ParsedQuery(
        original=original,
        semantic_remainder=_remove_spans(original, spans),
        geographies=tuple(item for item in geographies if id(item) in accepted_geo_ids),
        times=tuple(item for item in times if id(item) in accepted_time_ids),
    )
