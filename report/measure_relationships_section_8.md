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
- `outputs/relations/run_1b5f0cbf1084aaa8d017/relation_candidates.csv`
- `outputs/relations/run_1b5f0cbf1084aaa8d017/manual_relation_review_sample.csv`
- `outputs/relations/run_1b5f0cbf1084aaa8d017/relation_summary.json`

## Evaluation

`statvocab evaluate --area relations` checks endpoint validity, duplicate relation IDs, duplicate
pairs, self-relations, allowed relation types, and required evidence and confidence fields.

Manual review records whether each candidate is a valid relationship, whether the type is correct,
an optional corrected type, a false-positive category, and notes. When completed labels are
available, the evaluator reports precision@10, precision@25, precision@50, typed accuracy, and
false-positive taxonomy counts.

The refreshed deterministic export contains 5,628 candidate relationships:

| Type | Count |
|---|---:|
| `broader_than` | 442 |
| `related_to` | 5,184 |
| `variant_of` | 2 |

The structural validator passes: endpoints reference known final measures, relation IDs and pairs are
not duplicated, no self-relations are present, relation types are allowed, and evidence/confidence
fields are populated.

The refreshed manual review sample contains 100 candidates but has not yet been completed. Earlier
pre-refresh review metrics are retained only as historical development evidence; the formal milestone
9 report treats the refreshed graph as a structurally valid candidate output, not a fully adjudicated
semantic graph.

## Current Limitations

The first implementation does not try to exhaustively recover every possible semantic relation.
The current accepted export still includes many low-confidence `related_to` candidates, so it is
broader than a precision-oriented graph. The next improvement should complete the refreshed manual
review, filter accepted exports by confidence, and tighten hierarchy rules.
