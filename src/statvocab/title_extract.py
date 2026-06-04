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

    return cleaned_title, tuple(terms)
