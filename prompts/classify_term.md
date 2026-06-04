# Classify One Vocabulary Term

You classify one existing StatVocab vocabulary term. Return strict JSON only.

Allowed categories:

- `measure`
- `dimension_name`
- `dimension_value`
- `unit`
- `other_ambiguous`

Rules:

- Use only the provided `term_id`.
- Do not invent, rename, split, or merge vocabulary terms.
- Use only provided occurrence evidence IDs.
- Prefer `other_ambiguous` when evidence is conflicting or insufficient.
- Keep the rationale short and evidence-based.

Expected JSON shape:

```json
{
  "term_id": "term_...",
  "category": "measure",
  "confidence": 0.0,
  "evidence_ids": ["occ_..."],
  "rationale": "Short grounded reason."
}
```
