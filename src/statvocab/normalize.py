"""Conservative text normalization for extracted vocabulary."""

from __future__ import annotations

import re
import unicodedata

WHITESPACE_RE = re.compile(r"\s+")
TERMINAL_PUNCTUATION_RE = re.compile(r"[\s,;:.]+$")
NUMERIC_LIKE_RE = re.compile(
    r"^[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?\s*[A-Za-z]?$"
)
DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
    }
)


def normalize_display(value: str) -> str:
    """Return display-preserving normalized text."""

    normalized = unicodedata.normalize("NFKC", value)
    return WHITESPACE_RE.sub(" ", normalized).strip()


def normalize_matching_key(value: str) -> str:
    """Return a conservative deterministic matching key."""

    display = normalize_display(value)
    translated = display.translate(DASH_TRANSLATION)
    compact = WHITESPACE_RE.sub(" ", translated).strip().casefold()
    return TERMINAL_PUNCTUATION_RE.sub("", compact).strip()


def is_numeric_like(value: str) -> bool:
    """Return whether a metadata cell looks numeric or numeric-with-flag."""

    return bool(NUMERIC_LIKE_RE.match(value.strip()))


def collapse_removed_spans(value: str) -> str:
    """Clean whitespace and light punctuation left after span removal."""

    cleaned = WHITESPACE_RE.sub(" ", value)
    cleaned = re.sub(r"\s+([,;:])", r"\1", cleaned)
    cleaned = re.sub(r"([(\[{])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)\]}])", r"\1", cleaned)
    cleaned = re.sub(r"\s*,\s*(?:,|\)|$)", " ", cleaned)
    cleaned = re.sub(r"\(\s*\)", " ", cleaned)
    cleaned = re.sub(r"\[\s*\]", " ", cleaned)
    cleaned = re.sub(r"\s+-\s*(?:-|$)", " ", cleaned)
    return WHITESPACE_RE.sub(" ", cleaned).strip(" ,;:-")
