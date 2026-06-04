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

The completed deterministic sample contained 100 candidates ranked by confidence and stratified
across relation types. Review used a strict criterion: `related_to` required a useful semantic
connection beyond shared domain, geography, or breakdown wording; hierarchy required genuine
concept specialization; and fragments or dimension values were rejected even when classification
had placed them in the measure export.

Measured review results:

- precision@10: `1.00`
- precision@25: `0.96`
- precision@50: `0.90`
- precision over all 100 reviewed candidates: `0.78`
- typed accuracy among valid candidates: `0.821`

By proposed type, `variant_of` achieved `1.00` precision, `related_to` achieved `0.767`, and
`broader_than` achieved `0.706`. The main typing errors were presentation qualifiers incorrectly
treated as hierarchy, generic `related_to` labels where a clear hierarchy existed, and reordered or
singular/plural titles not recognized as variants.

The false-positive taxonomy contained:

- classification leakage from dimensions, units, or fragments: 8
- broad-domain similarity without a useful relation: 8
- shared geography, breakdowns, or qualifiers only: 4
- lexical overlap only: 1
- incomplete or generic term: 1

Candidates with confidence at least `0.70` achieved `0.909` precision in the reviewed sample,
compared with `0.78` over the unrestricted sample. This supports using `0.70` as a cautious accepted
export threshold while retaining lower-confidence candidates in the review artifact.

## Current Limitations

The first implementation does not try to exhaustively recover every possible semantic relation.
The current accepted export still includes low-confidence candidates, so it is broader than the
precision-oriented objective suggests. The reviewed sample supports filtering the accepted export
at confidence `0.70`, improving fragment detection before candidate generation, and tightening
hierarchy rules so temporal frequency and unit suffixes are not automatically treated as semantic
specialization.
