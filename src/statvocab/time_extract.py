"""Deterministic time expression extraction and interval normalization."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

from statvocab.config import ExtractionConfig
from statvocab.contracts import TimeGranularity, stable_id
from statvocab.normalize import normalize_display


@dataclass(frozen=True)
class TimeOccurrence:
    """One recognized time expression occurrence."""

    time_id: str
    table_id: str
    raw_value: str
    normalized_value: str
    start_date: str
    end_date: str
    granularity: str
    source_area: str
    location: str
    source_span_start: int | None
    source_span_end: int | None
    extractor_rule: str
    confidence: float


YEAR_RE = re.compile(r"^(?P<year>\d{4})(?:\.0)?$")
QUARTER_RE = re.compile(
    r"^(?:(?P<year_a>\d{4})[- ]?Q(?P<quarter_a>[1-4])|"
    r"Q(?P<quarter_b>[1-4])[- ]?(?P<year_b>\d{4}))$",
    re.IGNORECASE,
)
MONTH_CODE_RE = re.compile(r"^(?P<year>\d{4})[- ]?M(?P<month>0?[1-9]|1[0-2])$", re.IGNORECASE)
ISO_MONTH_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$")
ISO_DAY_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12]\d|3[01])$")

TITLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "title_iso_day",
        re.compile(r"\b(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12]\d|3[01])\b"),
    ),
    (
        "title_year_range",
        re.compile(
            r"\b(?P<start>\d{4})\s*(?:-|to|until|through)\s*(?P<end>\d{4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "title_until_year",
        re.compile(r"\b(?:until|up to)\s+(?P<year>\d{4})\b", re.IGNORECASE),
    ),
    (
        "title_quarter",
        re.compile(
            r"\b(?:(?P<year_a>\d{4})[- ]?Q(?P<quarter_a>[1-4])|"
            r"Q(?P<quarter_b>[1-4])[- ]?(?P<year_b>\d{4}))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "title_month_name",
        re.compile(
            r"\b(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<year>\d{4})\b",
            re.IGNORECASE,
        ),
    ),
    ("title_year", re.compile(r"\b(?P<year>\d{4})\b")),
)

MONTH_NAMES = {
    name.casefold(): index for index, name in enumerate(calendar.month_name) if name
}


def _date_string(year: int, month: int, day: int) -> str:
    return date(year, month, day).isoformat()


def _year_allowed(year: int, config: ExtractionConfig) -> bool:
    return config.min_year <= year <= config.max_year


def _year_interval(year: int) -> tuple[str, str, str, str]:
    return (
        str(year),
        _date_string(year, 1, 1),
        _date_string(year, 12, 31),
        TimeGranularity.YEAR.value,
    )


def _month_interval(year: int, month: int, normalized: str) -> tuple[str, str, str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return (
        normalized,
        _date_string(year, month, 1),
        _date_string(year, month, last_day),
        TimeGranularity.MONTH.value,
    )


def _quarter_interval(year: int, quarter: int) -> tuple[str, str, str, str]:
    start_month = ((quarter - 1) * 3) + 1
    end_month = start_month + 2
    last_day = calendar.monthrange(year, end_month)[1]
    return (
        f"{year}-Q{quarter}",
        _date_string(year, start_month, 1),
        _date_string(year, end_month, last_day),
        TimeGranularity.QUARTER.value,
    )


def _build_occurrence(
    *,
    table_id: str,
    raw_value: str,
    normalized_value: str,
    start_date: str,
    end_date: str,
    granularity: str,
    source_area: str,
    location: str,
    source_span_start: int | None,
    source_span_end: int | None,
    extractor_rule: str,
    confidence: float = 1.0,
) -> TimeOccurrence:
    time_id = stable_id("time", table_id, normalized_value, granularity)
    return TimeOccurrence(
        time_id=time_id,
        table_id=table_id,
        raw_value=raw_value,
        normalized_value=normalized_value,
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
        source_area=source_area,
        location=location,
        source_span_start=source_span_start,
        source_span_end=source_span_end,
        extractor_rule=extractor_rule,
        confidence=confidence,
    )


def parse_time_value(
    raw_value: str,
    *,
    table_id: str,
    source_area: str,
    location: str,
    config: ExtractionConfig,
    source_span_start: int | None = None,
    source_span_end: int | None = None,
    extractor_rule_prefix: str = "header",
) -> TimeOccurrence | None:
    """Parse one full time value into a normalized interval."""

    raw = normalize_display(raw_value)
    match = ISO_DAY_RE.match(raw)
    if match:
        year = int(match.group("year"))
        month = int(match.group("month"))
        day = int(match.group("day"))
        if not _year_allowed(year, config):
            return None
        normalized = f"{year:04d}-{month:02d}-{day:02d}"
        try:
            start = _date_string(year, month, day)
        except ValueError:
            return None
        return _build_occurrence(
            table_id=table_id,
            raw_value=raw_value,
            normalized_value=normalized,
            start_date=start,
            end_date=start,
            granularity=TimeGranularity.DAY.value,
            source_area=source_area,
            location=location,
            source_span_start=source_span_start,
            source_span_end=source_span_end,
            extractor_rule=f"{extractor_rule_prefix}_iso_day",
        )

    match = ISO_MONTH_RE.match(raw) or MONTH_CODE_RE.match(raw)
    if match:
        year = int(match.group("year"))
        month = int(match.group("month"))
        if not _year_allowed(year, config):
            return None
        normalized, start, end, granularity = _month_interval(
            year,
            month,
            f"{year:04d}-{month:02d}",
        )
        return _build_occurrence(
            table_id=table_id,
            raw_value=raw_value,
            normalized_value=normalized,
            start_date=start,
            end_date=end,
            granularity=granularity,
            source_area=source_area,
            location=location,
            source_span_start=source_span_start,
            source_span_end=source_span_end,
            extractor_rule=f"{extractor_rule_prefix}_month",
        )

    match = QUARTER_RE.match(raw)
    if match:
        year = int(match.group("year_a") or match.group("year_b"))
        quarter = int(match.group("quarter_a") or match.group("quarter_b"))
        if not _year_allowed(year, config):
            return None
        normalized, start, end, granularity = _quarter_interval(year, quarter)
        return _build_occurrence(
            table_id=table_id,
            raw_value=raw_value,
            normalized_value=normalized,
            start_date=start,
            end_date=end,
            granularity=granularity,
            source_area=source_area,
            location=location,
            source_span_start=source_span_start,
            source_span_end=source_span_end,
            extractor_rule=f"{extractor_rule_prefix}_quarter",
        )

    match = YEAR_RE.match(raw)
    if match:
        year = int(match.group("year"))
        if not _year_allowed(year, config):
            return None
        normalized, start, end, granularity = _year_interval(year)
        return _build_occurrence(
            table_id=table_id,
            raw_value=raw_value,
            normalized_value=normalized,
            start_date=start,
            end_date=end,
            granularity=granularity,
            source_area=source_area,
            location=location,
            source_span_start=source_span_start,
            source_span_end=source_span_end,
            extractor_rule=f"{extractor_rule_prefix}_year",
        )

    return None


def extract_header_times(
    table_id: str,
    time_columns: list[str],
    config: ExtractionConfig,
) -> list[TimeOccurrence]:
    """Extract time intervals from parsed time-header columns."""

    occurrences: list[TimeOccurrence] = []
    for index, value in enumerate(time_columns):
        parsed = parse_time_value(
            value,
            table_id=table_id,
            source_area="header_time",
            location=f"header[{index}]",
            config=config,
        )
        if parsed is not None:
            occurrences.append(parsed)
    return occurrences


def _overlaps(span: tuple[int, int], accepted: list[tuple[int, int]]) -> bool:
    return any(span[0] < other_end and other_start < span[1] for other_start, other_end in accepted)


def _title_match_to_occurrence(
    *,
    table_id: str,
    title: str,
    rule_id: str,
    match: re.Match[str],
    config: ExtractionConfig,
) -> TimeOccurrence | None:
    raw = match.group(0)
    start_span, end_span = match.span()

    if rule_id == "title_year_range":
        start_year = int(match.group("start"))
        end_year = int(match.group("end"))
        if end_year < start_year:
            return None
        if not (_year_allowed(start_year, config) and _year_allowed(end_year, config)):
            return None
        return _build_occurrence(
            table_id=table_id,
            raw_value=raw,
            normalized_value=f"{start_year}-{end_year}",
            start_date=_date_string(start_year, 1, 1),
            end_date=_date_string(end_year, 12, 31),
            granularity=TimeGranularity.RANGE.value,
            source_area="title",
            location="title",
            source_span_start=start_span,
            source_span_end=end_span,
            extractor_rule=rule_id,
            confidence=0.95,
        )

    if rule_id == "title_until_year":
        year = int(match.group("year"))
        if not _year_allowed(year, config):
            return None
        return _build_occurrence(
            table_id=table_id,
            raw_value=raw,
            normalized_value=f"until {year}",
            start_date=_date_string(config.min_year, 1, 1),
            end_date=_date_string(year, 12, 31),
            granularity=TimeGranularity.RANGE.value,
            source_area="title",
            location="title",
            source_span_start=start_span,
            source_span_end=end_span,
            extractor_rule=rule_id,
            confidence=0.9,
        )

    if rule_id == "title_month_name":
        year = int(match.group("year"))
        month = MONTH_NAMES[match.group("month").casefold()]
        if not _year_allowed(year, config):
            return None
        normalized, start, end, granularity = _month_interval(
            year,
            month,
            f"{year:04d}-{month:02d}",
        )
        return _build_occurrence(
            table_id=table_id,
            raw_value=raw,
            normalized_value=normalized,
            start_date=start,
            end_date=end,
            granularity=granularity,
            source_area="title",
            location="title",
            source_span_start=start_span,
            source_span_end=end_span,
            extractor_rule=rule_id,
            confidence=0.95,
        )

    return parse_time_value(
        raw,
        table_id=table_id,
        source_area="title",
        location=title,
        config=config,
        source_span_start=start_span,
        source_span_end=end_span,
        extractor_rule_prefix="title",
    )


def extract_title_times(
    table_id: str,
    title: str | None,
    config: ExtractionConfig,
) -> list[TimeOccurrence]:
    """Extract time intervals from a title, preferring longer deterministic spans."""

    if not title:
        return []

    occurrences: list[TimeOccurrence] = []
    accepted_spans: list[tuple[int, int]] = []
    for rule_id, pattern in TITLE_PATTERNS:
        for match in pattern.finditer(title):
            span = match.span()
            if _overlaps(span, accepted_spans):
                continue
            occurrence = _title_match_to_occurrence(
                table_id=table_id,
                title=title,
                rule_id=rule_id,
                match=match,
                config=config,
            )
            if occurrence is None:
                continue
            occurrences.append(occurrence)
            accepted_spans.append(span)
    return occurrences
