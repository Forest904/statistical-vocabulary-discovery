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
agreement, validation metrics, final-test metrics, and source-specific metrics for the random,
targeted reclaim, and compiled human-loop sources. The active validated artifact candidate remains
`run_75b90575bb9e48ce7918`.

Headline random-audit results:

| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| All random rows | 0.916 | 0.869 | 0.909 |
| Random validation | 0.930 | 0.881 | 0.916 |
| Random final test | 0.930 | 0.884 | 0.926 |

Targeted reclaim diagnostic stress test:

| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| All targeted rows | 0.204 | 0.121 | 0.196 |
| Targeted validation | 0.020 | 0.008 | 0.001 |
| Targeted final test | 0.100 | 0.051 | 0.164 |

The targeted reclaim audit covers 250 completed rows sampled from terms previously assigned to
`other_ambiguous`. It is reported as a diagnostic stress test rather than the headline submission
quality metric because it intentionally probes the accepted classifier's conservative abstention
behavior. The random duplicate relabel audit covers 50 rows, and the targeted duplicate relabel
audit covers 25 rows; both report raw agreement `1.000` and Cohen's kappa `1.000`. Accepted
hallucination count is `0`.

## Current Limitations

The 500 random primary labels, 50 random duplicate labels, 250 targeted reclaim labels, and 25
targeted duplicate labels are complete, but they were completed through repository audits rather
than a fully independent second-human annotation study. A gated reclaim run,
`run_5f8127cf8e6be829b1ac`, was not promoted because it exceeded the random final-test macro-F1
drop allowance (`0.036 > 0.020`), missed the targeted non-other precision floor
(`0.714 < 0.850`), and reduced `other_ambiguous` by only `0.007 < 0.200`. Accepted outputs remain
conservative, and no automatic `other_ambiguous` reduction is claimed. Human-loop review events are
treated as continuous ground-truth expansion for future classifier versions, not as a one-time gate.
