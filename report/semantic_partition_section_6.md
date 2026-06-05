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

The `local-hybrid` variant preserves protected rule outputs, caches PEARL-small embeddings, and
trains a calibrated local classifier from the completed audit labels. Low-confidence model outputs
abstain to `other_ambiguous`.

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

`report/classification_metrics.json` records completed audit-label metrics, duplicate relabel
agreement, validation metrics, and final-test metrics. The current local-hybrid run is
`run_75b90575bb9e48ce7918`.

Current results:

| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| All labeled rows | 0.948 | 0.920 | 0.940 |
| Validation | 0.940 | 0.892 | 0.926 |
| Final test | 0.960 | 0.927 | 0.954 |

The duplicate relabel audit covers 50 rows and reports raw agreement `1.000` and Cohen's kappa
`1.000`. Accepted hallucination count is `0`.

## Current Limitations

The 500 primary labels and 50 duplicate labels are complete, but they were completed through a
rule-assisted repository audit rather than a fully independent second-human annotation study. The
local classifier remains conservative: measure recall is the weakest final-test class, and many
uncertain measure-like terms are intentionally placed in `other_ambiguous`.
