# 8. Measure Relationships And Quality Evaluation

Milestone 5 generates cautious typed candidate relationships between final measure terms.

## Method

The relation command reads `outputs/measures.csv` and reuses the pinned PEARL-small embeddings
from semantic partitioning. If measure cluster assignments are available, their domain labels are
used as supporting evidence, but missing clustering artifacts do not block relationship generation.

Candidates combine embedding similarity, lexical containment, qualifier patterns, shared domains,
and title-family signals. Variant and hierarchy evidence takes precedence over generic semantic
similarity. Hierarchical candidates are exported as one directional edge from the broader measure to
the narrower measure. `narrower_than` is represented by reading `broader_than` in the inverse
direction rather than by duplicating the edge. Weak or contradictory candidates are kept out of the
accepted export.

LLM adjudication is scaffolded through `prompts/classify_relation.md`, but it is disabled by default
so the core command remains deterministic and reproducible.

## Outputs

The relation command writes:

- `outputs/measure_relations.csv`
- `outputs/relations/run_1b4d6642319427ea9aa0/relation_candidates.csv`
- `outputs/relations/run_1b4d6642319427ea9aa0/manual_relation_review_sample.csv`
- `outputs/relations/run_1b4d6642319427ea9aa0/relation_summary.json`

## Evaluation

`statvocab evaluate --area relations` checks endpoint validity, duplicate relation IDs, duplicate
pairs, self-relations, allowed relation types, and required evidence and confidence fields.

Manual review records whether each candidate is a valid relationship, whether the type is correct,
an optional corrected type, a false-positive category, and notes. When completed labels are
available, the evaluator reports precision@10, precision@25, precision@50, typed accuracy, and
false-positive taxonomy counts.

The submitted deterministic export contains 30,695 candidate relationships:

| Type | Count |
|---|---:|
| `broader_than` | 16,382 |
| `related_to` | 13,482 |
| `variant_of` | 831 |

The structural validator passes: endpoints reference known final measures, relation IDs and pairs are
not duplicated, no self-relations are present, relation types are allowed, and evidence/confidence
fields are populated.

The completed 100-row manual review sample recorded in `report/relations_metrics.json` reports
precision@10 `1.000`, precision@25 `1.000`, precision@50 `0.960`, precision@100 `0.890`, and typed
accuracy `0.888`. The false-positive taxonomy is dominated by generic qualifier and
denominator-fragment matches. Structural validation covers the refreshed 30,695-row export; the
current run's own review sample is available for the next review pass.

## Current Limitations

The first implementation does not try to exhaustively recover every possible semantic relation.
The current accepted export still includes broad `related_to` candidates and some lexical-containment
false positives, so the next improvement should tighten generic qualifier handling and hierarchy
rules.
