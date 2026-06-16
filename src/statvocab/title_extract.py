"""Title cleaning and conservative phrase generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from statvocab.contracts import stable_id
from statvocab.normalize import collapse_removed_spans, normalize_display, normalize_matching_key


@dataclass(frozen=True)
class TitleTerm:
    """One title-derived vocabulary candidate."""

    occurrence_id: str
    table_id: str
    raw_term: str
    normalized_term: str
    matching_key: str
    source_area: str
    source_span_start: int | None
    source_span_end: int | None
    extraction_rule: str
    is_full_title: bool


def remove_spans(title: str, spans: list[tuple[int, int]]) -> str:
    """Remove source spans from a title and clean leftover punctuation."""

    if not spans:
        return normalize_display(title)

    chars = list(title)
    for start, end in sorted(spans):
        for index in range(max(0, start), min(len(chars), end)):
            chars[index] = " "
    return collapse_removed_spans("".join(chars))


def _make_title_term(
    *,
    table_id: str,
    raw_term: str,
    source_area: str,
    source_span_start: int | None,
    source_span_end: int | None,
    extraction_rule: str,
    is_full_title: bool,
) -> TitleTerm | None:
    normalized = normalize_display(raw_term)
    matching_key = normalize_matching_key(normalized)
    if not matching_key:
        return None
    occurrence_id = stable_id(
        "occ",
        table_id,
        source_area,
        source_span_start,
        source_span_end,
        matching_key,
    )
    return TitleTerm(
        occurrence_id=occurrence_id,
        table_id=table_id,
        raw_term=raw_term,
        normalized_term=normalized,
        matching_key=matching_key,
        source_area=source_area,
        source_span_start=source_span_start,
        source_span_end=source_span_end,
        extraction_rule=extraction_rule,
        is_full_title=is_full_title,
    )


def split_conservative_clauses(cleaned_title: str) -> list[tuple[str, int, int, str]]:
    """Split title text only on high-confidence separators."""

    clauses: list[tuple[str, int, int, str]] = []
    for match in re.finditer(r"\s+-\s+|:\s+", cleaned_title):
        _ = match

    boundaries = [0]
    rules: list[str] = []
    for match in re.finditer(r"\s+-\s+|:\s+", cleaned_title):
        boundaries.append(match.start())
        rules.append("title_clause_separator")
        boundaries.append(match.end())
    boundaries.append(len(cleaned_title))

    if len(boundaries) == 2:
        return []

    pieces: list[tuple[int, int]] = []
    index = 0
    while index < len(boundaries) - 1:
        start = boundaries[index]
        end = boundaries[index + 1]
        if cleaned_title[start:end].strip():
            pieces.append((start, end))
        index += 2

    for start, end in pieces:
        text = cleaned_title[start:end].strip(" ,;:-")
        if text:
            clauses.append((text, start, end, "title_clause_separator"))

    for match in re.finditer(r"\(([^()]{3,})\)", cleaned_title):
        text = match.group(1).strip()
        if text:
            clauses.append((text, match.start(1), match.end(1), "title_parenthetical_clause"))

    return clauses


KEYPHRASE_STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "per",
    "the",
    "to",
    "with",
}
DIMENSION_VALUE_ONLY = {
    "annual",
    "monthly",
    "quarterly",
    "daily",
    "weekly",
    "total",
    "male",
    "female",
    "males",
    "females",
}
STATISTICAL_HEADS = {
    "area",
    "birth",
    "crime",
    "death",
    "employment",
    "expenditure",
    "gdp",
    "income",
    "index",
    "population",
    "price",
    "prices",
    "production",
    "rate",
    "sales",
    "turnover",
    "unemployment",
    "value added",
}


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _meaningful_tokens(value: str) -> list[str]:
    return [token for token in _tokens(value) if token not in KEYPHRASE_STOP_WORDS]


def _has_statistical_head(value: str) -> bool:
    lower = value.casefold()
    return any(head in lower for head in STATISTICAL_HEADS)


def _valid_keyphrase(value: str) -> bool:
    meaningful = _meaningful_tokens(value)
    if not 2 <= len(meaningful) <= 8:
        return False
    if set(meaningful) <= DIMENSION_VALUE_ONLY:
        return False
    return _has_statistical_head(value)


def _segment_keyphrases(cleaned_title: str) -> list[tuple[str, int, int, str]]:
    spans: list[tuple[str, int, int, str]] = []
    boundaries = [0]
    for match in re.finditer(r"\s+-\s+|[,;:()]|\b(?:by|of|in)\b", cleaned_title, re.IGNORECASE):
        boundaries.extend([match.start(), match.end()])
    boundaries.append(len(cleaned_title))

    for index in range(0, len(boundaries) - 1, 2):
        start = boundaries[index]
        end = boundaries[index + 1]
        text = cleaned_title[start:end].strip(" ,;:-()")
        if _valid_keyphrase(text):
            spans.append((text, start, end, "title_keyphrase_statistical_head"))
    return spans


def extract_title_keyphrases(cleaned_title: str) -> list[tuple[str, int, int, str]]:
    """Return deterministic statistical keyphrases from a cleaned title."""

    candidates = _segment_keyphrases(cleaned_title)
    words = list(re.finditer(r"[A-Za-z0-9][A-Za-z0-9'/-]*", cleaned_title))
    for start_index in range(len(words)):
        for end_index in range(start_index + 2, min(len(words), start_index + 8) + 1):
            start = words[start_index].start()
            end = words[end_index - 1].end()
            text = cleaned_title[start:end].strip(" ,;:-()")
            if _valid_keyphrase(text):
                candidates.append((text, start, end, "title_keyphrase_statistical_head"))

    seen: set[str] = set()
    unique: list[tuple[str, int, int, str]] = []
    for text, start, end, rule in candidates:
        key = normalize_matching_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((text, start, end, rule))
        if len(unique) >= 12:
            break
    return unique


def extract_title_terms(
    table_id: str,
    raw_title: str | None,
    removal_spans: list[tuple[int, int]],
) -> tuple[str, tuple[TitleTerm, ...]]:
    """Clean a title and return title-derived vocabulary candidates."""

    if not raw_title:
        return "", ()

    cleaned_title = remove_spans(raw_title, removal_spans)
    terms: list[TitleTerm] = []
    full = _make_title_term(
        table_id=table_id,
        raw_term=cleaned_title,
        source_area="title_full",
        source_span_start=None,
        source_span_end=None,
        extraction_rule="title_clean_full",
        is_full_title=True,
    )
    if full is not None:
        terms.append(full)

    seen = {full.matching_key} if full is not None else set()
    for text, start, end, rule in split_conservative_clauses(cleaned_title):
        term = _make_title_term(
            table_id=table_id,
            raw_term=text,
            source_area="title_clause",
            source_span_start=start,
            source_span_end=end,
            extraction_rule=rule,
            is_full_title=False,
        )
        if term is None or term.matching_key in seen:
            continue
        seen.add(term.matching_key)
        terms.append(term)

    for text, start, end, rule in extract_title_keyphrases(cleaned_title):
        term = _make_title_term(
            table_id=table_id,
            raw_term=text,
            source_area="title_keyphrase",
            source_span_start=start,
            source_span_end=end,
            extraction_rule=rule,
            is_full_title=False,
        )
        if term is None or term.matching_key in seen:
            continue
        seen.add(term.matching_key)
        terms.append(term)

    return cleaned_title, tuple(terms)
