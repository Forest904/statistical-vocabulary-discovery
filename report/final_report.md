---
title: "Statistical Vocabulary Discovery from Eurostat STAR Tables"
author: "Repository owner"
date: "2026-06-05"
---

# Submission Overview

This report follows the numbering of the STAR statistical-vocabulary assignment. The implementation targets the 2,000-table Eurostat STAR corpus, records resource provenance, produces the four required CSV files, and evaluates the semantic partition, clustering, and bonus relationships.

Final core run IDs:

| Stage | Run ID |
|---|---|
| Classification | `run_75b90575bb9e48ce7918` |
| Clustering | `run_eb9d50c4e3e315dafd9d` |
| Relationships | `run_1b5f0cbf1084aaa8d017` |
| Search index appendix | `run_c52b745f8105938306a7` |

Required assignment outputs:

| Output | Count |
|---|---:|
| `outputs/measures.csv` | 632 |
| `outputs/dimension_names.csv` | 365 |
| `outputs/dimension_values.csv` | 548 |
| `outputs/units.csv` | 214 |

Supplementary outputs:

| Output | Count |
|---|---:|
| `outputs/other_ambiguous.csv` | 7,812 |
| `outputs/measure_clusters.csv` | 632 |
| `outputs/measure_relations.csv` | 5,628 |

## Evidence Matrix

| Assignment step | Code | Notebook | Output evidence | Tests |
|---|---|---|---|---|
| 1. `D(t)` | `src/statvocab/time_extract.py`, `src/statvocab/title_extract.py` | `notebooks/02_time_geography_and_vocabulary.ipynb` | `data/processed/table_times.parquet`, `report/extraction_metrics.json` | `tests/test_time_extract.py` |
| 2. `S(t)` | `src/statvocab/ingest.py`, `src/statvocab/vocabulary.py` | `notebooks/01_ingestion_and_table_anatomy.ipynb`, `notebooks/02_time_geography_and_vocabulary.ipynb` | `data/processed/table_strings.parquet` | `tests/test_ingest.py`, `tests/test_observations.py` |
| 3. `Geo(t)` | `src/statvocab/geo_extract.py`, `src/statvocab/resources.py` | `notebooks/02_time_geography_and_vocabulary.ipynb` | `data/processed/table_geographies.parquet` | `tests/test_geo_extract.py` |
| 4. Titles | `src/statvocab/title_extract.py`, `src/statvocab/vocabulary.py` | `notebooks/02_time_geography_and_vocabulary.ipynb` | `data/processed/title_terms.parquet` | `tests/test_vocabulary.py` |
| 5. Global `V` | `src/statvocab/normalize.py`, `src/statvocab/vocabulary.py` | `notebooks/02_time_geography_and_vocabulary.ipynb` | `data/processed/vocabulary.parquet`, `data/processed/term_occurrences.parquet` | `tests/test_vocabulary.py` |
| 6. `M`, `N`, `A`, `U` | `src/statvocab/classification.py`, `src/statvocab/classify_rules.py`, `src/statvocab/classify_embeddings.py` | `notebooks/03_semantic_partition_and_evaluation.ipynb` | required CSVs, `report/classification_metrics.json` | `tests/test_classification_contracts.py`, `tests/test_grounding.py` |
| 7. Domains | `src/statvocab/cluster_measures.py` | `notebooks/04_measure_clustering.ipynb` | `outputs/measure_clusters.csv`, `report/clustering_metrics.json` | `tests/test_cluster_measures.py` |
| 8. Relationships | `src/statvocab/relations.py` | `notebooks/05_measure_relationships.ipynb` | `outputs/measure_relations.csv`, `report/relations_metrics.json` | `tests/test_relations.py` |

## Pipeline Diagram

```mermaid
flowchart LR
    A[Acquire resources] --> B[Ingest 2,000 STAR tables]
    B --> C[Extract D(t), S(t), Geo(t), title terms]
    C --> D[Build global vocabulary V]
    D --> E[Partition V into M, N, A, U, Other]
    E --> F[Cluster measures into domains]
    E --> G[Generate candidate relationships]
    E --> H[Build supplementary search indexes]
    F --> I[Notebooks and report]
    G --> I
    H --> I
```

# 1. Time Interval Extraction

The assignment asks for `D(t)`, the set of time intervals appearing on the top line of each table. The implementation reads the parsed table inventory, identifies the metadata/time boundary, and extracts years, quarters, months, dates, and ranges from time-header cells. Title time expressions are also recognized for title cleaning in step 4.

Each extracted time expression records the raw text, normalized interval, granularity, source location, rule ID, confidence, and table ID. The primary artifact is `data/processed/table_times.parquet`.

Evaluation is recorded in `report/extraction_metrics.json`. On the 50-table extraction review sample, time extraction has precision `1.000`, recall `1.000`, and F1 `1.000`. The limitation is deliberate conservatism: unsupported ambiguous periods are left unparsed rather than guessed.

# 2. Leftmost-Column String Extraction

The assignment asks for duplicate-free `S(t)`, the set of non-number strings in the leftmost columns, disjoint from `D(t)`. The ingestion stage detects contiguous metadata columns before time columns. The extraction stage reuses that boundary to collect metadata-column names and non-numeric metadata values while excluding numeric observations and observation flags.

The primary artifact is `data/processed/table_strings.parquet`. Ingestion inventoried exactly 2,000 tables; 1,771 parsed cleanly and 229 parsed with warnings, with zero failed tables.

Evaluation on the reviewed extraction sample reports string precision `1.000`, recall `1.000`, and F1 `1.000`. The main limitation is that malformed source rows are skipped consistently with ingestion diagnostics rather than repaired silently.

# 3. Geographic-Unit Identification

The assignment asks for `Geo(t) subset S(t)` using the Eurostat NUTS dictionary and then `V(t) = S(t) - Geo(t)`. The implementation supports two variants:

- `nuts`: assignment-compliant exact/normalized NUTS 2024 matching.
- `enhanced`: NUTS 2024 plus the official Eurostat `GEO` codelist.

The configured core run uses the enhanced variant for the submitted vocabulary while preserving the NUTS baseline in the extraction report. Accepted geography is excluded from `V(t)`; fuzzy geography is not accepted.

The primary artifact is `data/processed/table_geographies.parquet`. Evaluation on the reviewed sample reports geography precision `1.000`, recall `1.000`, and F1 `1.000`. Limitations remain around non-NUTS aggregates and geography expressions absent from the dictionaries.

# 4. Title Processing

The assignment asks to process each table title `tau(t)` like `S(t)`: separate dates and geography, then add the remainder to `V(t)`. The implementation removes recognized time and active-variant geography spans from titles, preserves the cleaned full title as strong measure evidence, and performs only conservative clause splitting.

The primary artifact is `data/processed/title_terms.parquet`. The method does not paraphrase titles or infer unstated concepts. This is important for grounding: title terms are source text, not generated labels.

Evaluation on the reviewed sample reports title precision `1.000`, recall `1.000`, and F1 `1.000`.

# 5. Global Vocabulary Construction

The assignment asks for the total vocabulary `V`, the distinct terms appearing in one or more tables after time and geography removal. The implementation combines metadata strings and cleaned title terms, applies display-preserving normalization, and generates stable term IDs and occurrence IDs.

Primary artifacts:

- `data/processed/vocabulary.parquet`
- `data/processed/term_occurrences.parquet`
- `data/processed/table_vocabulary.parquet`

Every global term has at least one source occurrence. Conflicting source roles are preserved in evidence summaries and later drive `other_ambiguous` decisions rather than being overwritten.

# 6. Vocabulary Partition and Quality Evaluation

The assignment asks to partition `V` into measures `M`, dimension names `N`, dimension values `A`, and unit information `U`. This project also uses `other_ambiguous` for terms with conflicting, fragmentary, or insufficient evidence.

The submitted local-hybrid classifier combines protected high-precision rules, structural features, PEARL-small embeddings, and a calibrated local classifier. Paid LLM adjudication is scaffolded through `prompts/classify_term.md`, but it is disabled by default and is not a core dependency. Accepted hallucination count is `0`.

Final output counts:

| Category | File | Count |
|---|---|---:|
| Measures `M` | `outputs/measures.csv` | 632 |
| Dimension names `N` | `outputs/dimension_names.csv` | 365 |
| Dimension values `A` | `outputs/dimension_values.csv` | 548 |
| Units `U` | `outputs/units.csv` | 214 |
| Other/ambiguous | `outputs/other_ambiguous.csv` | 7,812 |

Quality evaluation uses `data/gold/vocabulary_gold_labels.csv` with 500 completed audit labels and `data/gold/vocabulary_gold_relabel.csv` with 50 duplicate labels. The duplicate relabel audit reports raw agreement `1.000` and Cohen's kappa `1.000`.

Classification metrics from `report/classification_metrics.json`:

| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| All labeled rows | 0.948 | 0.920 | 0.940 |
| Validation | 0.940 | 0.892 | 0.926 |
| Final test | 0.960 | 0.927 | 0.954 |

The final-test measure class is the weakest class: F1 `0.667`, with measure recall `0.500`. This reflects the conservative design: uncertain measure-like terms are often moved into `other_ambiguous` instead of being forced into `M`.

Important limitation: the completed labels are a rule-assisted repository audit, not a fully independent second-human annotation study. The evaluation is still useful for checking consistency and class behavior, but the report does not claim independent human agreement beyond the duplicate audit file.

During the refreshed local-hybrid run, scikit-learn reported logistic-regression convergence warnings at the configured `max_iter=1000`. The run completed and validation passed, but this is another reason to treat the conservative `other_ambiguous` behavior as a known limitation rather than a fully optimized classifier.

# 7. Measure-Domain Clustering and Quality Evaluation

The assignment asks for an approach to group measures into domains. The implementation embeds the final measures with PEARL-small, clusters them with HDBSCAN, preserves HDBSCAN noise as `unclustered`, and assigns controlled domain labels using embedded domain descriptions and thresholds. Agglomerative clustering is exported as a baseline.

Current clustering metrics from `report/clustering_metrics.json`:

| Metric | Value |
|---|---:|
| Measure count | 632 |
| HDBSCAN cluster count | 35 |
| Agglomerative baseline clusters | 42 |
| Non-noise coverage | 0.574 |
| Unclustered measures | 269 |
| HDBSCAN stability proxy | 0.546 |

Domain distribution is concentrated in `cross-domain or other` because the domain labeler is conservative. Large visible domains include economy and finance, transport, labour market, agriculture, education, and industry/trade/services.

The refreshed manual cluster-review sample exists at `outputs/clustering/run_eb9d50c4e3e315dafd9d/manual_cluster_review_sample.csv`, but completed manual coherence labels are pending. The automatic validation passes: every cluster row references a valid measure, noise remains visible, and domain labels come from the controlled vocabulary.

# 8. Bonus Measure Relationships and Quality Evaluation

The assignment presents relationship discovery as bonus work. The implementation exports grounded candidate edges between final measures using embedding similarity, lexical containment, qualifier patterns, title-family signals, and shared domains.

Exported relation types:

- `broader_than`
- `variant_of`
- `related_to`

`narrower_than` is represented by reading a `broader_than` edge in the inverse direction; it is not duplicated as a separate row.

Current relationship metrics from `report/relations_metrics.json`:

| Metric | Value |
|---|---:|
| Candidate/exported relations | 5,628 |
| `broader_than` | 442 |
| `related_to` | 5,184 |
| `variant_of` | 2 |
| Validation passed | true |

Every relationship references known final measures, has evidence IDs, has confidence, and avoids self-relations and duplicate unordered pairs. LLM relationship adjudication is scaffolded through `prompts/classify_relation.md` but disabled by default.

The refreshed manual relation-review sample exists at `outputs/relations/run_1b5f0cbf1084aaa8d017/manual_relation_review_sample.csv`, but completed manual review labels are pending for this refreshed export. The output should therefore be read as a grounded candidate graph, not a fully adjudicated semantic graph.

# Appendices

## Resources and Licenses

Resources are recorded in `data/processed/resource_manifest.json` and summarized in `report/resource_register.md`. The core archive is `eurostat_2000_tables.tgz` from Zenodo record `15681384`, verified with MD5 `97bda36fec3c7c7ed042ee64ab16bf2c`. Other resources include the Eurostat title catalog, NUTS 2024 attributes, Eurostat `GEO` codelist, STAR questions/annotations, and PEARL-small.

Raw data, external downloads, caches, and generated heavy artifacts are intentionally excluded from git where appropriate. Acquisition scripts reproduce them with checksums.

## Prompts and Model IDs

Prompts are stored separately:

- `prompts/classify_term.md`
- `prompts/classify_relation.md`
- `prompts/label_cluster.md`

The local embedding model is `Lihuchen/pearl_small` at revision `0d29fb4a61ec2a11b60e8b078664389eb915286b`. Paid LLM calls are optional and disabled for the submitted core run.

## Annotation Guide and Agreement

The annotation guide is `report/annotation_guide.md`. The completed audit labels are in `data/gold/vocabulary_gold_labels.csv`; duplicate relabel rows are in `data/gold/vocabulary_gold_relabel.csv`.

Agreement: 50 duplicate relabel rows, raw agreement `1.000`, Cohen's kappa `1.000`. Because the milestone-9 labels were completed through a rule-assisted repository audit, this is reported as an internal consistency check rather than an independent multi-annotator study.

## Reproducibility Commands

Install:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks,ml,api]"
```

Run the core stages:

```bash
statvocab acquire --config configs/core.yaml
statvocab ingest --config configs/core.yaml
statvocab extract --config configs/core.yaml
statvocab classify --config configs/core.yaml --variant local-hybrid
statvocab cluster-measures --config configs/core.yaml
statvocab relations --config configs/core.yaml
statvocab build-search-index --config configs/core.yaml
```

Validate:

```bash
pytest tests/test_notebooks.py
statvocab validate-artifacts --run-id run_75b90575bb9e48ce7918
```

## Resource and Cost Summary

The core pipeline uses local Python processing and local PEARL-small embeddings. No paid API calls were used for the submitted core artifacts, so monetary LLM cost is `0`. Search index size for the refreshed appendix index is approximately 50 MB. The build uses CPU execution and cached model/resource files.

## Search Product Appendix

The journalism search product is supplementary. It demonstrates why the recovered vocabulary is useful, but it is not the central assignment deliverable. The search engine builds one document per source table, returns ranked Eurostat tables with evidence, and explicitly does not answer numeric questions.

Retrieval metrics are in `report/retrieval_metrics.json`; the refreshed index run is `run_c52b745f8105938306a7`.

Local demo:

```bash
docker compose up --build
```

## Scale Experiment Status

The 7,605-table full-corpus experiment is not attempted in this milestone. This is visible rather than hidden: the 2,000-table core corpus is the guaranteed release target, and full-corpus scalability remains future work.
