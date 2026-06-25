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
statvocab validate-artifacts --config configs/core.yaml --run-id run_42d33d7967059d3e8f22
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
statvocab evaluate --config configs/core.yaml --area extraction
statvocab evaluate --config configs/core.yaml --area classification
statvocab evaluate --config configs/core.yaml --area clustering
statvocab evaluate --config configs/core.yaml --area relations
statvocab evaluate --config configs/core.yaml --area retrieval
```

Final local validation:

```bash
pytest tests/test_notebooks.py
pytest tests/test_classification_contracts.py tests/test_grounding.py tests/test_relations.py tests/test_notebooks.py
statvocab validate-artifacts --config configs/core.yaml --run-id run_42d33d7967059d3e8f22
statvocab validate-core-release --config configs/core.yaml
```

## Current Core Outputs

| Output | Count | Role |
|---|---:|---|
| `outputs/measures.csv` | 2,894 | Required assignment `M` |
| `outputs/dimension_names.csv` | 359 | Required assignment `N` |
| `outputs/dimension_values.csv` | 607 | Required assignment `A` |
| `outputs/units.csv` | 261 | Required assignment `U` |
| `outputs/other_ambiguous.csv` | 9,114 | Justified extra category |
| `outputs/measure_clusters.csv` | 2,894 | Step 7 domain grouping |
| `outputs/measure_relations.csv` | 30,695 | Step 8 bonus candidate graph |
| `data/processed/knowledge_graph_nodes.parquet` | generated | Supplementary semantic graph nodes |
| `data/processed/knowledge_graph_edges.parquet` | generated | Supplementary semantic graph edges |

Submitted validated category export: `run_42d33d7967059d3e8f22`.
Latest local-hybrid rerun: `run_d96fee3c3c6166fddbac`; it validated but did not
replace the conservative submitted CSVs because the targeted reclaim promotion gate
did not improve. `report/core_release_manifest.json` is the frozen checksum snapshot
of the current professor-facing artifacts.

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

The repository also includes a grounded journalism-search product. It is supplementary
to the assignment submission and should be read after the assignment outputs and report
sections.

```bash
docker compose up --build
```

The FastAPI backend serves grounded search/evidence, artifact, relation, graph, and
review endpoints. The React frontend lets a user search for relevant Eurostat source
tables, inspect vocabulary and relation artifacts, explore bounded graph neighborhoods,
and perform human-loop review. The product does not answer numeric questions.

The supplementary knowledge graph stage materializes table, term, category, cluster, and domain nodes plus provenance-backed edges. The API serves bounded graph neighborhoods for the React Cytoscape.js explorer rather than rendering the full graph by default.

## Known Limitations

- Step-6 labels were completed through a rule-assisted repository audit; the duplicate relabel file is filled and agreement is reported, but it is not a fully independent second-human annotation study.
- The active classifier is intentionally conservative: uncertain or conflicting terms
  are placed in `other_ambiguous` rather than forced into required assignment files.
  The 250-row targeted reclaim audit is labeled and is used as future improvement data;
  the latest gated reclaim attempt did not satisfy the promotion gate.
- The refreshed clustering export is structurally validated against the current
  2,894-measure artifact set. The current 217-row manual review sample is complete:
  mean coherence is 1.226 on a 0-2 scale, 66.8% of reviewed assignments are at least
  coherent, and domain-label accuracy is 55.8%.
- The refreshed relationship export contains 30,695 structurally valid candidates. Its
  100-row manual review sample is completed and reported as a precision sample rather
  than an exhaustive guarantee.
- The 7,605-table full-corpus experiment is explicitly future scale work for cloud or
  larger local hardware. The submitted academic benchmark is the complete, validated
  2,000-table subset; no full-corpus semantic results are claimed.
