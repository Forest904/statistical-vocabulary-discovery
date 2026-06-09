---
title: "STAR Statistic Vocabularies Parsing and Structuring"
author: "Luca Foresti"
date: "2026-06-09"
---

# Repository Details & Overview

This report follows the numbering of the STAR statistical-vocabulary assignment and documents the current implementation on the 2,000-table Eurostat STAR subset. All quantitative results in the main report are from `configs/core.yaml`, whose configured corpus is `eurostat_2000_tables.tgz` with expected table count 2,000. The larger 7,605-table corpus is treated as pending scale work and is not used for the reported figures.

Repository link: `https://github.com/Forest904/statistical-vocabulary-discovery`

Installation is documented in `README.md`. The core setup is:

```bash
git clone https://github.com/Forest904/statistical-vocabulary-discovery.git
cd statistical-vocabulary-discovery
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks,ml,api]"
```

External resources are recorded in `data/processed/resource_manifest.json` and summarized in `report/resource_register.md`. They include the STAR 2,000-table archive from Zenodo record `15681384`, the table-title catalog, Eurostat NUTS 2024 geography data, the Eurostat `GEO` codelist used by the enhanced geography variant, STAR retrieval resources, and the PEARL-small embedding model. Prompt templates are stored separately in `prompts/classify_term.md`, `prompts/label_cluster.md`, and `prompts/classify_relation.md`. Paid LLM calls are optional and disabled for the submitted core run.

Final core run IDs:

| Stage | Run ID |
|---|---|
| Classification | `run_75b90575bb9e48ce7918` |
| Targeted reclaim proposal | `run_2b1f232c08e3f03b0deb` |
| Clustering | `run_eb9d50c4e3e315dafd9d` |
| Relationships | `run_e6e523656ea9857ecac9` |
| Search appendix | `run_c52b745f8105938306a7` |

Required assignment outputs:

| Output | Count |
|---|---:|
| `outputs/measures.csv` | 632 |
| `outputs/dimension_names.csv` | 365 |
| `outputs/dimension_values.csv` | 548 |
| `outputs/units.csv` | 214 |

Supplementary outputs:

| Output | Count | Role |
|---|---:|---|
| `outputs/other_ambiguous.csv` | 7,812 | Justified extra category for unsafe terms |
| `outputs/measure_clusters.csv` | 632 | Step 7 domain grouping |
| `outputs/measure_relations.csv` | 1,719 | Step 8 filtered bonus candidate graph |

# 1. Extraction of Time Intervals, `D(t)`

Methodology. The extractor reads the parsed table inventory, uses the detected metadata/time-column boundary, and scans time-header cells on the top line of each table. It recognizes years, integer-like year values such as `2024.0`, quarters, month labels, ISO-style dates, explicit year ranges, and selected title time phrases needed for title cleaning. Rough pattern matching is intentionally preferred over inference: for example, year-like values are accepted only when they satisfy configured bounds and context checks.

Data granularity. Each accepted expression is normalized to a typed interval with raw text, start date, end date, granularity, source area, source location or title span, rule ID, confidence, and deterministic `time_id`. Supported granularities include years, quarters, months, days, and ranges.

Output and evaluation. The primary artifact is `data/processed/table_times.parquet`. On the 50-table extraction review sample, `report/extraction_metrics.json` reports time precision `1.000`, recall `1.000`, and F1 `1.000`.

Limitations. Ambiguous or unsupported period expressions are left unparsed rather than guessed. This makes errors visible in review instead of silently adding questionable time intervals.

Code reference. Temporal extraction is implemented in `src/statvocab/time_extract.py`; title-time removal used in step 4 is implemented in `src/statvocab/title_extract.py`. Regression coverage is in `tests/test_time_extract.py`.

# 2. Extraction of Leftmost Column Strings, `S(t)`

Methodology. The ingestion stage detects contiguous metadata columns before the time columns. String extraction reopens each parsed source table from `parsed_local_path`, collects metadata-column names as header evidence, and collects non-blank, non-missing, non-numeric metadata values as candidate strings. Observation cells and one-letter observation flags are excluded.

Constraints implementation. `S(t)` is duplicate-free because terms are normalized to matching keys before table-level aggregation. It is disjoint from `D(t)` because recognized time-header values are routed to the time artifact and are not emitted as leftmost-column vocabulary strings.

Output and evaluation. The primary artifact is `data/processed/table_strings.parquet`. Ingestion inventoried exactly 2,000 tables: 1,771 parsed cleanly, 229 parsed with warnings, and zero failed. On the reviewed extraction sample, string precision, recall, and F1 are all `1.000`.

Limitations. Malformed rows are skipped consistently with ingestion diagnostics rather than repaired silently. This favors reproducibility over speculative table repair.

Code reference. Table parsing and metadata-boundary handling are in `src/statvocab/ingest.py`; vocabulary occurrence construction is in `src/statvocab/vocabulary.py`. Relevant tests include `tests/test_ingest.py` and `tests/test_observations.py`.

# 3. Identification of Geographical Units, `Geo(t)`

Dictionary integration. The assignment-compliant geography baseline uses Eurostat NUTS 2024 code and name matching. The submitted core configuration also supports an enhanced variant that adds the official Eurostat `GEO` codelist. Accepted matches are exact raw matches, exact normalized matches, or official alias matches; fuzzy geography matches are not accepted.

Vocabulary construction. For each table, accepted geography terms are removed from leftmost-column strings before vocabulary construction, implementing `V(t) = S(t) \setminus Geo(t)`. The configured core run uses the enhanced geography variant for submitted vocabulary files while preserving the NUTS-only baseline as an auditable method variant.

Output and evaluation. The primary artifact is `data/processed/table_geographies.parquet`. On the reviewed extraction sample, geography precision, recall, and F1 are all `1.000`.

Limitations. Non-NUTS aggregates and geography expressions absent from the official dictionaries may remain unresolved. The implementation does not invent geography labels to cover those cases.

Code reference. NUTS and Eurostat `GEO` matching are implemented in `src/statvocab/geo_extract.py`; resource acquisition and provenance are handled by `src/statvocab/resources.py`. Regression coverage is in `tests/test_geo_extract.py`.

# 4. Processing Table Titles, `tau(t)`

Title retrieval. Table titles are acquired from the provided natural-language title catalog and joined to parsed table IDs during ingestion and vocabulary construction.

Entity separation. Title processing removes recognized time spans and active-variant geography spans using the same conservative temporal and geographic evidence used for table cells. The cleaned full title is preserved as strong measure evidence. Clause splitting is limited to high-confidence separators and parenthetical expressions.

Vocabulary expansion. Cleaned title terms are appended to each table vocabulary `V(t)` after time and geography removal. The process preserves source text and does not paraphrase, rename, split aggressively, or infer concepts not present in the title.

Output and evaluation. The primary artifact is `data/processed/title_terms.parquet`. On the reviewed extraction sample, title precision, recall, and F1 are all `1.000`.

Limitations. Because the method avoids paraphrase, some semantically useful title variants are not generated. This is a deliberate grounding choice.

Code reference. Title cleaning and title-term construction are implemented in `src/statvocab/title_extract.py` and integrated into table vocabulary construction in `src/statvocab/vocabulary.py`. Tests are in `tests/test_vocabulary.py`.

# 5. Construction of Total Vocabulary, `V`

Aggregation. The total vocabulary is built as the union of all table vocabularies after removing time and geography evidence: `V = union_t V(t)`. The implementation combines non-temporal, non-geographic metadata strings with cleaned title terms.

Optimization. Terms are deduplicated with display-preserving normalization: matching keys casefold text, normalize whitespace, normalize compatible dash and apostrophe forms, and strip harmless terminal punctuation while preserving the original display text. Stable term IDs and occurrence IDs are generated from source evidence. This prevents repeated classification work for terms that recur across many tables.

Outputs. The primary artifacts are `data/processed/vocabulary.parquet` for distinct global terms, `data/processed/term_occurrences.parquet` for provenance, and `data/processed/table_vocabulary.parquet` for per-table vocabulary membership.

Limitations. Conflicting source roles are preserved in evidence summaries rather than overwritten. Later stages may route those terms to `other_ambiguous` instead of forcing a category.

Code reference. Normalization is implemented in `src/statvocab/normalize.py`; vocabulary aggregation is implemented in `src/statvocab/vocabulary.py`. Regression coverage is in `tests/test_vocabulary.py`.

# 6. Vocabulary Categorization and Partitioning

Partitioning methodology. The required partition divides `V` into measures `M`, dimension names `N`, dimension values `A`, and unit information `U`. The submitted `local-hybrid` classifier combines protected high-precision rules, structural evidence features, PEARL-small embeddings, and a calibrated local classifier trained on completed audit labels. Low-confidence or conflicting model outputs abstain to `other_ambiguous`.

Extended category. `other_ambiguous` is used because the assignment allows an extra category when needed. It contains fragmentary terms, conflicting evidence cases, and terms whose category would otherwise require unsafe inference. This category is not one of the four required output files, but it prevents low-confidence terms from polluting `M`, `N`, `A`, or `U`.

LLM guardrails. Optional paid LLM adjudication is scaffolded through `prompts/classify_term.md`, but it is disabled by default and is not a dependency of the submitted core artifacts. When enabled, the architecture requires strict JSON schema validation, existing `term_id` membership, evidence-ID membership, and category values from the controlled label set. The submitted run has accepted hallucination count `0`.

Artifact mapping:

| Category | File | Count |
|---|---|---:|
| Measures `M` | `outputs/measures.csv` | 632 |
| Dimension names `N` | `outputs/dimension_names.csv` | 365 |
| Dimension values `A` | `outputs/dimension_values.csv` | 548 |
| Units `U` | `outputs/units.csv` | 214 |
| Other/ambiguous | `outputs/other_ambiguous.csv` | 7,812 |

Quality evaluation. Evaluation uses `data/gold/vocabulary_gold_labels.csv` with 500 completed audit labels and `data/gold/vocabulary_gold_relabel.csv` with 50 duplicate labels. The duplicate relabel audit reports raw agreement `1.000` and Cohen's kappa `1.000`. Metrics from `report/classification_metrics.json` are:

| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| All labeled rows | 0.948 | 0.920 | 0.940 |
| Validation | 0.940 | 0.892 | 0.926 |
| Final test | 0.960 | 0.927 | 0.954 |

The final-test measure class is the weakest class: F1 `0.667`, with measure recall `0.500`. This reflects a conservative design choice: uncertain measure-like terms are often placed in `other_ambiguous` instead of being forced into `M`.

Targeted reclaim audit. To address the large `other_ambiguous` bucket without overclaiming, the repository now generates a separate human-only targeted review: `data/gold/vocabulary_reclaim_sample.csv` and `data/gold/vocabulary_reclaim_labels.csv` contain 250 candidate terms, with 25 blind duplicate rows in `data/gold/vocabulary_reclaim_relabel.csv`. The sample targets title terms, title clauses, frequent metadata values, unit/range/digit metadata values, short codes, conflicts, and noise-like terms. These labels are blank at the time of this report, so the original submitted CSVs remain the accepted outputs. A gated proposed classifier run, `run_2b1f232c08e3f03b0deb`, uses balanced class weights, `max_iter=5000`, and validation-threshold selection, but writes only proposed CSVs until targeted final-test labels are completed and the acceptance gate passes.

Limitations. The completed labels were produced through a rule-assisted repository audit rather than a fully independent second-human annotation study. The targeted reclaim audit is prepared but not yet completed, so no reduction of `other_ambiguous` is claimed here.

Code reference. Partition orchestration and acceptance gating are in `src/statvocab/classification.py`; deterministic rules are in `src/statvocab/classify_rules.py`; embedding support is in `src/statvocab/classify_embeddings.py`; targeted review sampling is in `src/statvocab/targeted_review.py`; optional LLM adjudication is in `src/statvocab/classify_llm.py`. Tests include `tests/test_classification_contracts.py`, `tests/test_classify_rules.py`, and `tests/test_grounding.py`.

# 7. Structuring and Clustering Measures

Clustering approach. Step 7 starts from the final measures in `outputs/measures.csv`. Measures are embedded with the same pinned PEARL-small configuration used in classification, then clustered with HDBSCAN. HDBSCAN noise points are preserved as `unclustered` so unstable assignments remain visible rather than being forced into a domain.

Knowledge sourcing. Domain labels come from a controlled top-level domain vocabulary. For each non-noise cluster, representative measures nearest the cluster centroid are compared with embedded domain descriptions. A domain is assigned only when the best score clears the configured threshold and margin; otherwise the cluster is labeled `cross-domain or other`. Agglomerative clustering with cosine distance and average linkage is exported as a baseline.

Outputs. The main clustering artifact is `outputs/measure_clusters.csv`. Supporting artifacts are in `outputs/clustering/run_eb9d50c4e3e315dafd9d/`: `agglomerative_baseline.csv`, `domain_taxonomy.json`, `manual_cluster_review_sample.csv`, and `clustering_summary.json`.

Quality evaluation. Current metrics from `report/clustering_metrics.json` are:

| Metric | Value |
|---|---:|
| Measures | 632 |
| HDBSCAN clusters | 35 |
| Agglomerative baseline clusters | 42 |
| Non-noise coverage | 0.574 |
| Unclustered measures | 269 |
| HDBSCAN stability proxy | 0.546 |
| Validation passed | true |

Domain distribution is conservative: `cross-domain or other` contains 414 measures. Large visible domains include economy and finance, transport, labour market, agriculture, education, and industry/trade/services.

Manual review status. The 100-row manual cluster-review template at `outputs/clustering/run_eb9d50c4e3e315dafd9d/manual_cluster_review_sample.csv` uses fixed fields for coherence score, domain-label quality, representative quality, and notes. The evaluator now reports mean coherence, coherent fraction, domain-label accuracy, representative quality distribution, and note counts once those fields are completed.

Limitations. The manual cluster-review sample has not yet been completed, so human coherence and label-quality scores are pending. The current quality surface is automatic validation, coverage, domain distribution, representative checks, and the visible unclustered set.

Code reference. Measure clustering, baseline export, domain labeling, and manual-review sampling are implemented in `src/statvocab/cluster_measures.py`. Manual-review metric parsing is implemented in `src/statvocab/clustering_evaluate.py`; tests are in `tests/test_cluster_measures.py`.

# 8. Semantic Relationships Between Measures (Bonus)

Relationship identification. Step 8 is implemented as a bonus candidate graph over final measures. Candidates are generated from embedding similarity, lexical containment, qualifier patterns, title-family signals, and shared domain evidence. Variant and hierarchy evidence take precedence over generic relatedness.

Exported relationship types:

| Type | Count |
|---|---:|
| `broader_than` | 442 |
| `related_to` | 1,275 |
| `variant_of` | 2 |

`narrower_than` is represented by reading a `broader_than` edge in the inverse direction, so it is not duplicated as a separate row in `outputs/measure_relations.csv`.

Quality evaluation. `report/relations_metrics.json` reports 1,719 submitted candidate relationships and successful structural validation. The full 5,628 accepted pre-filter candidates are preserved in `outputs/relations/run_e6e523656ea9857ecac9/accepted_relations_full.csv`, while the submitted `outputs/measure_relations.csv` keeps all hierarchy/variant edges and only `related_to` edges with confidence at least `0.70`. Every submitted relation references known final measures, avoids self-relations, avoids duplicate unordered pairs, has an allowed relation type, and includes evidence and confidence fields. Submitted confidence bands are 1,510 candidates from `0.70` to `0.84` and 209 from `0.85` to `1.00`.

LLM guardrails. Optional LLM relationship adjudication is scaffolded through `prompts/classify_relation.md` and disabled by default. The prompt requires existing endpoint IDs, controlled relation types, evidence-ID membership, self-relation rejection, and strict JSON.

Manual review status. The 100-row manual relation-review template is `outputs/relations/run_e6e523656ea9857ecac9/manual_relation_review_sample.csv`. It uses fixed fields for validity, type correctness, optional corrected type, false-positive type, and notes. Once completed, `src/statvocab/relations_evaluate.py` reports precision@10, precision@25, precision@50, precision@100, typed accuracy, and a false-positive taxonomy.

Limitations. The refreshed manual relation-review sample has not yet been completed. The output should therefore be read as a structurally valid, confidence-filtered, grounded candidate graph rather than a fully adjudicated semantic taxonomy.

Code reference. Candidate generation, confidence-filtered export, and full-candidate preservation are implemented in `src/statvocab/relations.py`. Evaluation is implemented in `src/statvocab/relations_evaluate.py`; tests are in `tests/test_relations.py`.

# Appendices

## A. Resources and Licenses

Resources are recorded in `data/processed/resource_manifest.json` and summarized in `report/resource_register.md`. The core archive is `eurostat_2000_tables.tgz` from Zenodo record `15681384`, verified with MD5 `97bda36fec3c7c7ed042ee64ab16bf2c`. Other resources include the Eurostat title catalog, NUTS 2024 attributes, the Eurostat `GEO` codelist, STAR questions and annotations, and PEARL-small.

Raw data, external downloads, caches, and generated heavy artifacts are intentionally excluded from git where appropriate. Acquisition commands reproduce them with recorded provenance.

## B. Prompts and Model IDs

Prompt files are stored separately:

| Prompt | Purpose |
|---|---|
| `prompts/classify_term.md` | Optional vocabulary-term adjudication |
| `prompts/label_cluster.md` | Optional constrained cluster-domain labeling |
| `prompts/classify_relation.md` | Optional relationship adjudication |

The local embedding model is `Lihuchen/pearl_small`; the exact pinned revision is recorded in `configs/core.yaml`. No paid API calls were used for the submitted core artifacts, so monetary LLM cost is `0`.

## C. Annotation Guide and Agreement

The annotation guide is `report/annotation_guide.md`. Completed vocabulary audit labels are in `data/gold/vocabulary_gold_labels.csv`; duplicate relabel rows are in `data/gold/vocabulary_gold_relabel.csv`.

Agreement on the 50 duplicate relabel rows is raw agreement `1.000` and Cohen's kappa `1.000`. Because the labels were completed through a rule-assisted repository audit, this is reported as an internal consistency check rather than an independent multi-annotator study.

The targeted reclaim review files are generated but not completed: `data/gold/vocabulary_reclaim_labels.csv` contains 250 rows and `data/gold/vocabulary_reclaim_relabel.csv` contains 25 duplicate relabel rows.

## D. Reproducibility Commands

Run the core stages:

```bash
statvocab acquire --config configs/core.yaml
statvocab ingest --config configs/core.yaml
statvocab extract --config configs/core.yaml
statvocab prepare-targeted-review --config configs/core.yaml
statvocab classify --config configs/core.yaml --variant local-hybrid
statvocab cluster-measures --config configs/core.yaml
statvocab relations --config configs/core.yaml
statvocab build-search-index --config configs/core.yaml
```

Validate the submitted artifacts:

```bash
pytest tests/test_notebooks.py
statvocab validate-artifacts --run-id run_75b90575bb9e48ce7918
```

## E. Supplementary Search and Knowledge Graph

The search and knowledge-graph features are supplementary and not central assignment deliverables. The search engine builds one document per source table, returns ranked Eurostat tables with evidence, and does not answer numeric questions. The knowledge graph materializes bounded table, term, category, cluster, and domain neighborhoods for inspection.

Retrieval metrics are in `report/retrieval_metrics.json`; the refreshed search-index run is `run_c52b745f8105938306a7`.

Local demo command:

```bash
docker compose up --build
```

## F. Full-Scale Run Status

The full-scale 7,605-table run is pending and no full-scale results are claimed in this report. The 2,000-table subset is the completed, validated submission target. Full-scale figures and bottleneck analysis should be added only after the large run finishes and its artifacts are validated.
