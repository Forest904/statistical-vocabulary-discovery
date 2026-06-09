# Statistical Vocabulary Discovery

Interpretable extraction, classification, and semantic structuring of vocabularies from Eurostat-style statistical tables.

This repository is organized as an academic submission for the STAR statistical-vocabulary assignment. The core release targets the 2,000-table Eurostat corpus and produces the required vocabulary files:

- `outputs/measures.csv`
- `outputs/dimension_names.csv`
- `outputs/dimension_values.csv`
- `outputs/units.csv`

Supplementary outputs include `outputs/other_ambiguous.csv`, `outputs/measure_clusters.csv`, `outputs/measure_relations.csv`, notebooks, and a formal report in `report/final_report.md` / `report/final_report.pdf`.

## Assignment Reading Order

1. Read `report/final_report.pdf` or `report/final_report.md`.
2. Run or inspect notebooks in filename order from `notebooks/00_project_and_data_card.ipynb` through `notebooks/07_final_results_reproducibility_and_limitations.ipynb`.
3. Validate the required CSVs with:

```bash
statvocab validate-artifacts --run-id run_75b90575bb9e48ce7918
```

The report follows the assignment numbering exactly:

1. time interval extraction `D(t)`
2. leftmost-column string extraction `S(t)`
3. geographic units `Geo(t)` and `V(t) = S(t) - Geo(t)`
4. title processing
5. global vocabulary `V`
6. partition into measures `M`, dimension names `N`, dimension values `A`, and units `U`
7. measure-domain clustering
8. bonus measure relationships

## Installation

Python 3.12 is required.

```bash
git clone <repository-url>
cd statistical-vocabulary-discovery

python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks,ml,api]"
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

For the React demo:

```bash
npm --prefix web install
```

## Reproduction

The pipeline stages can be run explicitly:

```bash
statvocab acquire --config configs/core.yaml
statvocab ingest --config configs/core.yaml
statvocab extract --config configs/core.yaml
statvocab prepare-targeted-review --config configs/core.yaml
statvocab classify --config configs/core.yaml --variant local-hybrid
statvocab cluster-measures --config configs/core.yaml
statvocab relations --config configs/core.yaml
statvocab build-knowledge-graph --config configs/core.yaml
statvocab build-search-index --config configs/core.yaml
statvocab evaluate --config configs/evaluation.yaml --area extraction
statvocab evaluate --config configs/evaluation.yaml --area clustering
statvocab evaluate --config configs/evaluation.yaml --area relations
statvocab evaluate --config configs/evaluation.yaml --area retrieval
```

Final local validation:

```bash
pytest tests/test_notebooks.py
pytest tests/test_classification_contracts.py tests/test_grounding.py tests/test_relations.py tests/test_notebooks.py
statvocab validate-artifacts --run-id run_75b90575bb9e48ce7918
```

## Current Core Outputs

| Output | Count | Role |
|---|---:|---|
| `outputs/measures.csv` | 632 | Required assignment `M` |
| `outputs/dimension_names.csv` | 365 | Required assignment `N` |
| `outputs/dimension_values.csv` | 548 | Required assignment `A` |
| `outputs/units.csv` | 214 | Required assignment `U` |
| `outputs/other_ambiguous.csv` | 7,812 | Justified extra category |
| `outputs/measure_clusters.csv` | 632 | Step 7 domain grouping |
| `outputs/measure_relations.csv` | 1,719 | Step 8 filtered bonus candidates |
| `data/processed/knowledge_graph_nodes.parquet` | generated | Supplementary semantic graph nodes |
| `data/processed/knowledge_graph_edges.parquet` | generated | Supplementary semantic graph edges |

Final classification run: `run_75b90575bb9e48ce7918`.

## Pipeline Diagram

```mermaid
flowchart LR
    A[Acquire resources] --> B[Ingest tables]
    B --> C[Extract D(t), S(t), Geo(t), titles]
    C --> D[Build global V]
    D --> E[Partition into M, N, A, U, Other]
    E --> F[Cluster measures]
    E --> G[Generate measure relationships]
    G --> H[Build knowledge graph]
    E --> I[Build search documents and indexes]
    F --> I[Report and notebooks]
    H --> I
```

## Architecture Diagram

```mermaid
flowchart TB
    subgraph Sources
        S1[STAR Eurostat tables]
        S2[Eurostat titles]
        S3[NUTS 2024 and Eurostat GEO]
        S4[STAR questions and annotations]
    end
    subgraph Python Package
        P1[resources.py / ingest.py]
        P2[time_extract.py / geo_extract.py / vocabulary.py]
        P3[classification.py]
        P4[cluster_measures.py / relations.py / knowledge_graph.py]
        P5[search engine]
    end
    subgraph Submission
        O1[Required CSVs]
        O2[Metrics JSON]
        O3[Notebooks]
        O4[Formal report]
    end
    Sources --> Python Package
    Python Package --> Submission
```

## Product Demo

The repository also includes a minimal journalism-search product. It is supplementary to the assignment submission.

```bash
docker compose up --build
```

The FastAPI backend serves grounded search/evidence endpoints, and the React frontend lets a user search for relevant Eurostat source tables. The product does not answer numeric questions.

The supplementary knowledge graph stage materializes table, term, category, cluster, and domain nodes plus provenance-backed edges. The API serves bounded graph neighborhoods for the React Cytoscape.js explorer rather than rendering the full graph by default.

## Known Limitations

- Step-6 labels were completed through a rule-assisted repository audit; the duplicate relabel file is filled and agreement is reported, but it is not a fully independent second-human annotation study.
- The local classifier is conservative: many uncertain measure-like terms are placed in `other_ambiguous`. The 250-row targeted reclaim audit is now labeled, but the latest gated classifier attempt did not satisfy the promotion gate and remains in a proposed run directory.
- The refreshed clustering and relationship exports are structurally validated, and their 100-row manual review samples are completed in the current report metrics.
- The 7,605-table full-corpus scale experiment is not yet attempted in this milestone.
