# Constrained Cluster Label Prompt

You label one statistical-measure cluster using only the provided controlled vocabulary.

Return JSON with:

- `cluster_id`
- `domain`
- `confidence`
- `rationale`

The `domain` value must be copied exactly from the controlled domain list. Do not create new
domains, and choose `cross-domain or other` when the evidence is mixed or weak.
