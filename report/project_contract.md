# Project Contract Mapping

`ai/PRD.md` is the local product and research contract for StatVocab. The `ai/`
directory remains ignored, so this tracked document records the public contract
surface that later milestones must satisfy.

## Assignment Step Mapping

| Step | Requirement | Planned modules | Planned artifacts | Notebook/report surface |
|---|---|---|---|---|
| 1 | Extract time intervals `D(t)` | `time_extract.py`, `title_extract.py` | `table_times.parquet` | Notebook 02, report section 1 |
| 2 | Extract duplicate-free left-column strings `S(t)` | `ingest.py`, `vocabulary.py` | `table_strings.parquet` | Notebook 02, report section 2 |
| 3 | Identify geography and construct `V(t)` | `geo_extract.py`, `resources.py` | `table_geographies.parquet` | Notebook 02, report section 3 |
| 4 | Process titles before adding title vocabulary | `title_extract.py`, `vocabulary.py` | `title_terms.parquet` | Notebook 02, report section 4 |
| 5 | Build global vocabulary `V` | `normalize.py`, `vocabulary.py` | `vocabulary.parquet` | Notebook 02, report section 5 |
| 6 | Partition `V` into semantic categories | `classify_rules.py`, `classify_embeddings.py`, `classify_hybrid.py`, `classify_llm.py` | required CSVs in `outputs/` | Notebook 03, report section 6 |
| 7 | Group measures into domains | `cluster_measures.py` | `measure_clusters.csv` | Notebook 04, report section 7 |
| 8 | Identify typed measure relationships | `relations.py` | `measure_relations.csv` | Notebook 05, report section 8 |

## Foundation Decisions

- Python runtime is pinned to Python 3.12.
- Production logic belongs in `src/statvocab/`; notebooks explain and orchestrate.
- The core release targets the 2,000-table STAR Eurostat corpus.
- The 7,605-table corpus is a scale experiment, not a core-release blocker.
- Search returns grounded source tables and evidence, never numeric answers.
- LLM output is optional, cached, schema-validated, and constrained to known terms.

