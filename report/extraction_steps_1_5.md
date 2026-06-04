# Assignment Steps 1-5: Extraction Draft

This draft documents Milestone 2 outputs produced by:

```bash
statvocab extract --config configs/core.yaml
statvocab evaluate --config configs/evaluation.yaml --area extraction
```

## 1. Time Interval Extraction

The extractor reads Milestone 1 `tables.parquet`, uses detected time-header
columns, and also scans linked titles. It recognizes years, integer-like years
such as `2024.0`, quarters, months, ISO dates, explicit year ranges, and
`until YYYY` title phrases. Each occurrence is normalized to a typed interval
with start date, end date, granularity, raw expression, source area, source
location or title span, rule ID, confidence, and deterministic `time_id`.

Output: `data/processed/table_times.parquet`.

Limitations: ambiguous period phrases are not guessed. They should appear in
review failures rather than being silently normalized.

## 2. Leftmost-Column String Extraction

The extractor reopens each parsed source CSV from `parsed_local_path`, using
the Milestone 1 metadata/time column boundary. Metadata column names become
header evidence; non-blank, non-missing, non-numeric metadata values become
candidate strings. Malformed rows are skipped consistently with ingestion, and
observation cells never enter the vocabulary.

Output: `data/processed/table_strings.parquet`.

## 3. Geographic-Unit Identification

Two geography variants are recorded:

- `nuts`: assignment-compliant NUTS 2024 code/name matching.
- `enhanced`: NUTS 2024 plus the official Eurostat `GEO` codelist when
  acquired locally.

Matching is exact raw, exact normalized, or official alias matching only. Fuzzy
matches are not accepted and do not remove terms from `V(t)`. The configured
active variant, currently `enhanced`, controls which accepted geography is
excluded from vocabulary.

Output: `data/processed/table_geographies.parquet`.

## 4. Title Processing

Titles are cleaned by removing recognized time and active-variant geography
spans. The cleaned full title is preserved as strong measure evidence. Clause
splitting is conservative and limited to high-confidence separators and
parenthetical expressions; no paraphrases are generated.

Output: `data/processed/title_terms.parquet`.

## 5. Global Vocabulary Construction

The global vocabulary is built from non-temporal, non-geographic metadata
strings and cleaned title terms. Normalization preserves display text while
matching keys casefold and normalize whitespace, compatible dash/apostrophe
forms, and harmless terminal punctuation. Stable occurrence IDs and term IDs
are generated from source evidence and matching keys.

Outputs:

- `data/processed/term_occurrences.parquet`
- `data/processed/table_vocabulary.parquet`
- `data/processed/vocabulary.parquet`

Every global term has at least one occurrence, and role conflicts are preserved
in JSON summaries rather than overwritten.

## Evaluation Draft

`statvocab evaluate --area extraction` writes:

- `data/processed/extraction_review_sample.csv`
- `data/processed/extraction_gold_template.csv`
- `report/extraction_metrics.json`

The tracked fixture corpus includes `extraction_gold.csv` for automated
regression tests. The 50-table core review remains a human annotation workflow:
the template is exported first, and metrics are computed when labels are
completed.
