# Classify One Measure Relationship Candidate

You adjudicate one existing StatVocab measure relationship candidate. Return strict JSON only.

Allowed relation types:

- `broader_than`
- `narrower_than`
- `variant_of`
- `related_to`

Rules:

- Use only the provided `source_term_id` and `target_term_id`.
- Do not invent, rename, split, or merge measure terms.
- Use only provided evidence IDs.
- Reject the candidate when evidence is insufficient, endpoints are invalid, or the relation would be a self-relation.
- Keep the rationale short and grounded in the supplied evidence.

Expected JSON shape:

```json
{
  "source_term_id": "term_...",
  "target_term_id": "term_...",
  "relation_type": "related_to",
  "accept": true,
  "confidence": 0.0,
  "evidence_ids": ["occ_..."],
  "rationale": "Short grounded reason."
}
```
