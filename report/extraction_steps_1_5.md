# Assignment Steps 1-5: Milestone 2 Extraction

This report documents the completed Milestone 2 extraction pipeline for the
core 2,000-table Eurostat STAR corpus. The current artifacts are produced by:

```bash
statvocab extract --config configs/core.yaml
statvocab evaluate --config configs/core.yaml --area extraction
```

## 1. Time Interval Extraction

The extractor reads Milestone 1 `tables.parquet`, uses detected time-header
columns, and scans linked titles. It recognizes years, integer-like years such
as `2024.0`, standard quarters, French quarter labels found in the core corpus
such as `Janvier-Mars 1995`, months, ISO dates, explicit year ranges, and
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

## Evaluation And Gold Review

`statvocab evaluate --area extraction` writes:

- `data/processed/extraction_review_sample.csv`
- `data/processed/extraction_gold_template.csv`
- `report/extraction_metrics.json`

The completed 50-table human review is tracked separately from generated
processed artifacts:

- `data/gold/extraction_review_sample.csv`
- `data/gold/extraction_gold_template.csv`
- `data/gold/extraction_gold_labels.csv`

The tracked fixture corpus still includes `extraction_gold.csv` for automated
regression tests. For core/full evaluation, `extraction_gold_labels.csv` in
`data/gold/` is the reviewed ground truth used for metrics.

Current core review metrics:

| Area | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time | 128 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| String | 170 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Geography | 97 | 0 | 0 | 22 | 1.000 | 1.000 | 1.000 |
| Title | 50 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |

The review sample is intentionally scoped rather than exhaustive. Remaining
limitations are deterministic rather than silent: unsupported time expressions
outside the implemented rules are left unparsed, geography matching remains
exact dictionary matching, and title processing does not paraphrase or infer
concepts beyond cleaned title evidence.
