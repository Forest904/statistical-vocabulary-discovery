# Full-Corpus Scalability And Bottleneck Report

Full-corpus run ID: `run_6c602fe0d70beb33cf3c`

## Stage Summary

| Stage | Status | Seconds | Peak RSS bytes | Artifact bytes | Blocker |
|---|---:|---:|---:|---:|---|
| disk-preflight | succeeded | 0.16 | 71049216 | 0 |  |
| acquire | succeeded | 1305.56 | 84725760 | 5390 |  |
| ingest | failed | 8833.80 |  | 0 | 1 validation error for ParseWarning reason   String should have at least 1 character [type=string_too_short, input_value='', input_type=str]     For further information visit https://errors.pydantic.dev/2.13/v/string_too_short |
| extract | skipped | 0.00 |  | 0 | ingestion did not complete in this run |
| classify-local-hybrid | skipped | 0.00 |  | 0 | extraction did not complete in this run |
| cluster-measures | skipped | 0.00 |  | 0 | classification did not complete in this run |
| relations | skipped | 0.00 |  | 0 | classification did not complete in this run |
| build-search-index | skipped | 0.00 |  | 0 | classification did not complete in this run |
| evaluate-retrieval | skipped | 0.00 |  | 0 | classification did not complete in this run |

## Bottleneck Policy

No algorithmic optimization is applied by this orchestration layer. Stages marked failed, blocked, or unusually expensive in the table above are the evidence source for any later optimization.
