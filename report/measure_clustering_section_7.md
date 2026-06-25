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
- `outputs/clustering/run_af85e9400741e5dc8d39/agglomerative_baseline.csv`
- `outputs/clustering/run_af85e9400741e5dc8d39/domain_taxonomy.json`
- `outputs/clustering/run_af85e9400741e5dc8d39/manual_cluster_review_sample.csv`
- `outputs/clustering/run_af85e9400741e5dc8d39/clustering_summary.json`

## Evaluation

`statvocab evaluate --area clustering` reports coverage, cluster-size distribution, domain
distribution, representative counts, silhouette where valid, a HDBSCAN stability proxy, baseline
cluster counts, and manual-review completion status.

Current refreshed run:

| Metric | Value |
|---|---:|
| Run ID | `run_af85e9400741e5dc8d39` |
| Measures | 2,894 |
| HDBSCAN clusters | 216 |
| Agglomerative baseline clusters | 48 |
| Non-noise coverage | 0.870 |
| Unclustered measures | 376 |
| HDBSCAN stability proxy | 0.828 |
| Manual review rows completed | 217 |
| Mean manual coherence | 1.226 / 2 |
| Coherent or strongly coherent | 0.668 |
| Domain-label accuracy | 0.558 |
| Representative good fraction | 0.839 |

Manual review records coherence, domain-label quality, representative quality, and error examples.
The current refreshed review sample is complete in `report/clustering_metrics.json`: 121 domain
labels were marked correct, 24 partial, and 72 wrong; 182 representatives were marked good and
35 poor.

## Current Limitations

The first implementation prioritizes reproducible artifacts and explicit review data over tuned
cluster quality. The refreshed export has high structural coverage but still leaves 376 measures
unclustered and places many rows in `cross-domain or other`. The manual review suggests that the
clearest next improvement is tuning domain-label thresholds and cross-domain fallbacks from this
evidence.
