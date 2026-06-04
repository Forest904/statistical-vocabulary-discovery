# StatVocab Annotation Guide

This guide defines the mutually exclusive Milestone 3 labels for global vocabulary terms.
Annotate the term as it appears in the sampled row, using the term text plus its source-role
evidence. Time expressions and accepted geography should already be excluded from global `V`.

## Labels

- `measure`: a statistical phenomenon or quantity being measured. Examples: `Pesticide sales by categorisation of active substances`, `Agricultural land prices`, `Value added, gross`.
- `dimension_name`: the name of an axis, variable, or metadata column that organizes observations. Examples: `Unit of measure`, `Time frequency`, `Sex`, `Geopolitical entity (reporting)\Time`.
- `dimension_value`: a value belonging to a dimension. Examples: `Annual`, `females`, `From 15 to 24 years`, `Total economy`.
- `unit`: a measurement unit, scale, rate denominator, or unit-like expression. Examples: `Percentage`, `Million euro`, `Thousand persons`, `Per hundred thousand inhabitants`.
- `other_ambiguous`: a term whose source evidence is conflicting, insufficient, malformed, or too broad to assign safely.

## Tie-Breaking Rules

- Prefer source role over surface text when they conflict.
- Header-only terms are usually `dimension_name`.
- Metadata values are usually `dimension_value`, unless they are clear units.
- Title-only full or clause terms are usually `measure` when they name a phenomenon.
- A raw term that appears in incompatible roles should be `other_ambiguous` unless one role is clearly erroneous.
- Use `other_ambiguous` for uncertain cases rather than forcing a category.

## Required Notes

Annotator notes are required when:

- category is `other_ambiguous`;
- a term could reasonably be both a measure and a dimension value;
- a unit-like term is used as a dimension value;
- source-role evidence conflicts across tables;
- the term is a code, abbreviation, or fragment that cannot be interpreted confidently.

## Blind Relabel And Adjudication

At least 10% of the sample is duplicated in `vocabulary_gold_relabel.csv`.
The duplicate labeler should not inspect the primary label. Evaluation reports raw agreement,
Cohen's kappa, disagreements, and adjudication status. Final adjudicated labels remain in
`vocabulary_gold_labels.csv`.

## LLM Guarding

LLM suggestions are optional and untrusted. They may only reference known `term_id` values,
allowed category labels, and occurrence evidence belonging to the same term. Malformed JSON,
unknown terms, unknown evidence, or invented categories must be rejected.
