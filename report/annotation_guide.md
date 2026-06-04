# Annotation Guide Outline

## Category Labels

- `measure`: a statistical quantity or phenomenon being measured.
- `dimension_name`: a variable or axis name that organizes observations.
- `dimension_value`: a value belonging to a dimension.
- `unit`: measurement unit or unit-like scale information.
- `other_ambiguous`: justified ambiguous, conflicting-role, or unsupported terms.

## Relation Labels

- `broader_than`: source measure is a wider concept than the target measure.
- `narrower_than`: source measure is a narrower concept than the target measure.
- `variant_of`: measures describe near-equivalent concepts with qualifiers.
- `related_to`: measures are meaningfully associated but not hierarchical.

## Grounding Rules

- Do not label time expressions or accepted geography as vocabulary terms.
- Preserve conflicting-role evidence instead of forcing a category.
- Prefer `other_ambiguous` when evidence is insufficient.
- Require annotator notes for uncertain, borderline, or conflicting cases.
- Treat LLM suggestions as untrusted until schema and membership checks pass.

## Planned Examples

Later milestones will add positive, negative, and borderline examples from the
fixture corpus and from the stratified gold vocabulary sample.

