# 7. Measure-Domain Clustering And Quality Evaluation

Milestone 4 groups final measure terms into statistical domains while preserving terms that should
not be forced into a cluster.

## Method

The clustering command reads only `outputs/measures.csv`, reuses the pinned PEARL-small embedding
configuration from semantic partitioning, and clusters normalized embeddings with HDBSCAN. HDBSCAN
noise points are exported as `unclustered` so review can see the measures that do not fit a stable
local neighborhood.

Each non-noise cluster receives representative measures nearest the cluster centroid. Domain labels
are selected from the controlled top-level vocabulary defined in the project PRD. The local labeling
path compares cluster evidence to embedded domain descriptions and assigns a domain only when the
best score clears the configured threshold and margin; otherwise the cluster is labeled
`cross-domain or other`.

The required baseline is agglomerative clustering with cosine distance and average linkage. Its
assignments are exported alongside the HDBSCAN artifacts for comparison.

## Outputs

The clustering command writes:

- `outputs/measure_clusters.csv`
- `outputs/clustering/<run_id>/agglomerative_baseline.csv`
- `outputs/clustering/<run_id>/domain_taxonomy.json`
- `outputs/clustering/<run_id>/manual_cluster_review_sample.csv`
- `outputs/clustering/<run_id>/clustering_summary.json`

## Evaluation

`statvocab evaluate --area clustering` reports coverage, cluster-size distribution, domain
distribution, representative counts, silhouette where valid, a HDBSCAN stability proxy, baseline
cluster counts, and manual-review completion status.

Manual review records coherence, domain-label quality, representative quality, and error examples.
Until review fields are completed, the automatic metrics are available but human coherence and label
quality remain pending.

## Current Limitations

The first implementation prioritizes reproducible artifacts and explicit review data over tuned
cluster quality. Thresholds and HDBSCAN parameters should be revisited after the manual review sample
has enough completed rows to judge the coverage/coherence tradeoff.
