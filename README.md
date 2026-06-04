# statistical-vocabulary-discovery

Interpretable extraction, classification, and semantic structuring of vocabularies from Eurostat-style statistical tables.

## Overview

This project explores how to recover meaningful vocabularies from statistical tables whose semantics are distributed across titles, row headers, column headers, dates, geographic labels, measures, dimension names, dimension values, and units.

The goal is to build a reproducible pipeline that extracts table vocabularies, removes time and geographic entities, partitions the remaining terms into semantic categories, clusters statistical measures into domains, and experiments with semantic relationships between measures.

## Goals

* Extract time intervals from table headers.
* Extract non-numeric vocabulary terms from table rows and titles.
* Detect and remove geographic entities using external geographic dictionaries such as NUTS.
* Partition vocabulary terms into:

  * measures
  * dimension names
  * dimension values
  * units
  * optional other/ambiguous terms
* Cluster measures into semantic domains.
* Experiment with measure-to-measure semantic relationships.
* Evaluate extraction, classification, clustering, and relationship quality.

## Planned Approach

The project uses a hybrid, traceable methodology:

1. **Rule-based extraction** for dates, numeric regions, text headers, and units.
2. **Layout-aware features** to distinguish dimension names from dimension values.
3. **Dictionary matching** for geographic entities.
4. **Embedding-based similarity** for ambiguous terms and measure clustering.
5. **Optional constrained LLM adjudication** for difficult cases, with guardrails to prevent hallucinated vocabulary.
6. **Manual evaluation** on sampled terms and relationships.

## Repository Structure

```text
statistical-vocabulary-discovery/
  data/
    raw/                  # input datasets, not committed
    external/             # NUTS/geographic dictionaries, taxonomies
    processed/            # intermediate artifacts, not committed
  outputs/
    measures.csv
    dimension_names.csv
    dimension_values.csv
    units.csv
    other.csv
    measure_clusters.csv
    measure_relations.csv
  prompts/
    classify_term.md
    classify_relation.md
  src/statvocab/
    ingest.py
    table_layout.py
    normalize.py
    time_extract.py
    geo_extract.py
    vocab_extract.py
    classify_rules.py
    classify_embeddings.py
    classify_llm.py
    cluster_measures.py
    relations.py
    evaluate.py
    cli.py
  tests/
  notebooks/
  report/
```

## Installation

```bash
git clone https://github.com/<your-username>/statistical-vocabulary-discovery.git
cd statistical-vocabulary-discovery

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

## Usage

The command-line interface is under active development. Planned commands:

```bash
# Extract vocabulary from raw statistical tables
statvocab extract --input data/raw --output data/processed/vocabulary.parquet

# Classify vocabulary terms
statvocab classify --input data/processed/vocabulary.parquet --output outputs/

# Cluster extracted measures
statvocab cluster-measures --input outputs/measures.csv --output outputs/measure_clusters.csv

# Discover candidate semantic relationships between measures
statvocab relations --input outputs/measures.csv --output outputs/measure_relations.csv

# Run evaluation
statvocab evaluate --gold data/processed/gold_labels.csv --pred outputs/
```

## Outputs

The repository will produce the required assignment files:

```text
outputs/measures.csv
outputs/dimension_names.csv
outputs/dimension_values.csv
outputs/units.csv
```

Additional experimental outputs:

```text
outputs/other.csv
outputs/measure_clusters.csv
outputs/measure_relations.csv
```

## Evaluation

The project will evaluate:

* semantic category classification using accuracy, macro-F1, and per-class precision/recall;
* measure clustering using manual cluster coherence and domain-purity checks;
* semantic relationships using precision@k over manually reviewed candidate edges.

