"""Interpretable high-precision semantic classification rules."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from statvocab.contracts import VocabularyCategory


@dataclass(frozen=True)
class RuleDecision:
    """A single rule classification decision."""

    category: VocabularyCategory
    confidence: float
    evidence: str
    protected: bool


UNIT_EXACT = {
    "%",
    "% gdp",
    "euro",
    "million euro",
    "thousand euro",
    "kilogram",
    "tonne",
    "tonnes",
    "thousand tonnes",
    "hectare",
    "hour",
    "person",
    "persons",
    "thousand persons",
    "inhabitants",
    "per hundred thousand inhabitants",
    "percentage",
    "percentage of gross domestic product (gdp)",
    "number",
    "rate",
    "index",
}
UNIT_PATTERNS = (
    r"\bper\s+(?:hundred|thousand|million)\b",
    r"\b(?:million|thousand|billion)\s+(?:euro|persons?|tonnes?|inhabitants?)\b",
    r"\bindex\b.*\b(?:=|average)\b",
    r"\bpercentage\b",
    r"^%(\s|$)",
)
DIMENSION_NAME_HINTS = (
    "unit of measure",
    "time frequency",
    "geopolitical entity",
    "classification",
    "category",
    "categorisation",
    "sex",
    "age class",
    "age group",
    "economic activity",
    "statistical classification",
)
FREQUENCY_VALUES = {"annual", "monthly", "quarterly", "daily", "weekly", "year"}
DIMENSION_VALUE_HINTS = (
    "total",
    "males",
    "females",
    "male",
    "female",
    "less than",
    "from ",
    "years or over",
    "all isced",
    "unadjusted data",
    "employed persons",
    "unemployed persons",
)
MEASURE_TITLE_HINTS = (
    "population",
    "employment",
    "unemployment",
    "production",
    "prices",
    "price",
    "sales",
    "income",
    "expenditure",
    "risk indicator",
    "value added",
    "gross domestic product",
    "gdp",
    "birth",
    "death",
    "crime",
    "transport",
    "pesticide",
    "agricultural",
)


def _json_dict(value: object) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, dict) else {}


def _is_unit(term: str, key: str) -> bool:
    if key in UNIT_EXACT:
        return True
    return any(re.search(pattern, term.casefold()) for pattern in UNIT_PATTERNS)


def _is_age_band(term: str) -> bool:
    lower = term.casefold()
    return bool(re.search(r"\b\d+\s*(?:to|-)\s*\d+\s*years?\b", lower)) or (
        "years or over" in lower
    )


def classify_feature_row(row: dict[str, Any]) -> RuleDecision:
    """Classify one feature row using protected, readable rules."""

    term = str(row["canonical_term"]).strip()
    key = str(row["matching_key"]).strip().casefold()
    lower = term.casefold()
    roles = _json_dict(row.get("role_summary_json"))
    has_header = bool(row.get("has_header_evidence"))
    has_title = bool(row.get("has_title_evidence"))
    has_value = bool(row.get("has_metadata_value_evidence"))
    has_conflict = bool(row.get("has_conflicting_roles"))

    if not term:
        return RuleDecision(
            VocabularyCategory.OTHER_AMBIGUOUS,
            1.0,
            "blank term rejected by safety rule",
            True,
        )

    if has_header and not has_value and not has_title:
        return RuleDecision(
            VocabularyCategory.DIMENSION_NAME,
            0.98,
            "metadata column name evidence",
            True,
        )

    if any(hint in lower for hint in DIMENSION_NAME_HINTS) and has_header:
        return RuleDecision(
            VocabularyCategory.DIMENSION_NAME,
            0.96,
            "dimension-name dictionary matched metadata header evidence",
            True,
        )

    if _is_unit(term, key):
        return RuleDecision(
            VocabularyCategory.UNIT,
            0.96,
            "unit dictionary or unit lexical pattern matched",
            True,
        )

    if key in FREQUENCY_VALUES or _is_age_band(term):
        return RuleDecision(
            VocabularyCategory.DIMENSION_VALUE,
            0.94,
            "frequency or age-band dimension value pattern matched",
            True,
        )

    if has_conflict or len(roles) > 1:
        return RuleDecision(
            VocabularyCategory.OTHER_AMBIGUOUS,
            0.70,
            "term has conflicting source-role evidence",
            False,
        )

    if has_value and any(hint in lower for hint in DIMENSION_VALUE_HINTS):
        return RuleDecision(
            VocabularyCategory.DIMENSION_VALUE,
            0.90,
            "metadata value dictionary matched",
            True,
        )

    if has_title and any(hint in lower for hint in MEASURE_TITLE_HINTS):
        return RuleDecision(
            VocabularyCategory.MEASURE,
            0.88,
            "title evidence with measure lexical cue",
            True,
        )

    occurrence_count = int(row.get("occurrence_count") or 0)
    if has_title and not has_value and not has_header and occurrence_count == 1:
        return RuleDecision(
            VocabularyCategory.MEASURE,
            0.72,
            "title-only term treated as likely measure",
            False,
        )

    return RuleDecision(
        VocabularyCategory.OTHER_AMBIGUOUS,
        0.55,
        "no high-precision rule matched",
        False,
    )
