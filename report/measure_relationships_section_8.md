# 8. Measure Relationships And Quality Evaluation

Milestone 5 generates cautious typed candidate relationships between final measure terms.

## Method

The relation command reads `outputs/measures.csv` and reuses the pinned PEARL-small embeddings
from semantic partitioning. If measure cluster assignments are available, their domain labels are
used as supporting evidence, but missing clustering artifacts do not block relationship generation.

Candidates combine embedding similarity, lexical containment, qualifier patterns, shared domains,
and title-family signals. Variant and hierarchy evidence takes precedence over generic semantic
similarity. Hierarchical candidates are exported as one directional edge from the broader measure to
the narrower measure. Weak or contradictory candidates are kept out of the accepted export.

LLM adjudication is scaffolded through `prompts/classify_relation.md`, but it is disabled by default
so the core command remains deterministic and reproducible.

## Outputs

The relation command writes:

- `outputs/measure_relations.csv`
- `outputs/relations/<run_id>/relation_candidates.csv`
- `outputs/relations/<run_id>/manual_relation_review_sample.csv`
- `outputs/relations/<run_id>/relation_summary.json`

## Evaluation

`statvocab evaluate --area relations` checks endpoint validity, duplicate relation IDs, duplicate
pairs, self-relations, allowed relation types, and required evidence and confidence fields.

Manual review records whether each candidate is a valid relationship, whether the type is correct,
an optional corrected type, a false-positive category, and notes. When completed labels are
available, the evaluator reports precision@10, precision@25, precision@50, typed accuracy, and
false-positive taxonomy counts.

## Current Limitations

The first implementation is precision-oriented and does not try to exhaustively recover every
possible semantic relation. Embedding thresholds and qualifier lists should be revisited after the
manual review sample has enough completed rows to expose the dominant false-positive patterns.
