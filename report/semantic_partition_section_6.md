# 6. Vocabulary Partition And Quality Evaluation

Milestone 3 partitions the global vocabulary `V` into the required assignment categories:
measures, dimension names, dimension values, and units, with `other_ambiguous` for justified
unsafe cases.

## Method

The implementation reads `data/processed/vocabulary.parquet` and aggregates structural evidence
from source roles, table counts, occurrence counts, metadata columns, title evidence, and header
evidence. A deterministic annotation sampler writes a stratified 500-term gold template and a
10% blind relabel template.

The `rule-only` variant applies protected high-precision rules for clear metadata column names,
unit expressions, frequency and age-band values, common dimension values, and title-backed
measures. Every term receives exactly one category; unmatched or conflicting cases become
`other_ambiguous`.

The `local-hybrid` variant preserves protected rule outputs. When completed train/development
labels and ML dependencies are available, it caches PEARL-small embeddings and trains a calibrated
local classifier. Until gold labels are filled, it records that local modeling is pending and falls
back to the rule partition.

Optional paid adjudication is scaffolded through `prompts/classify_term.md`, strict schema
validation, term-membership checks, and evidence-membership checks. Paid calls are disabled by
default.

## Outputs

The classification command writes:

- `data/gold/vocabulary_gold_sample.csv`
- `data/gold/vocabulary_gold_labels.csv`
- `data/gold/vocabulary_gold_relabel.csv`
- `data/processed/classification_features.parquet`
- `outputs/classification/<run_id>/<variant>_predictions.parquet`
- `outputs/measures.csv`
- `outputs/dimension_names.csv`
- `outputs/dimension_values.csv`
- `outputs/units.csv`
- `outputs/other_ambiguous.csv`

## Evaluation

`statvocab evaluate --area classification` reports gold-label status, agreement status, and metrics
when labels are complete. Metrics include accuracy, macro-F1, weighted-F1, per-class scores,
confusion matrix, validation metrics, and final-test metrics. The selected primary variant must be
chosen by validation macro-F1, subject to zero accepted hallucinations, without using final-test
labels.

## Current Limitations

The first implementation pass creates the templates and runnable classification pipeline, but the
Milestone 3 exit gate remains incomplete until the 500-term gold labels and blind relabel rows are
manually completed and final-test results are documented.
