# Initial Resource Register

| Resource | Role | Milestone | Versioning requirement |
|---|---|---:|---|
| STAR Eurostat 2,000-table archive | Core corpus | 1 | DOI, filename, size, MD5 `97bda36fec3c7c7ed042ee64ab16bf2c` |
| STAR Eurostat 7,605-table archive | Scale corpus | 10 | DOI, filename, size, MD5 `175222a2497fbceee6f9f6ede6728e9c` |
| Assignment title catalog | Table title evidence | 1 | URL, checksum, retrieval timestamp |
| Eurostat NUTS 2024 | Assignment geography dictionary | 1 | exact release/download timestamp and checksum |
| Eurostat code lists | Enhanced geography and terminology | 2 | exact resources, timestamps, checksums |
| STAR supplementary repository | Retrieval questions, annotations, baselines | 1 | pinned commit SHA and file checksums |
| `S_i.csv` | Lexically close retrieval questions | 6 | source URL and checksum |
| `S_r.csv` | Semantically reformulated questions | 6 | source URL and checksum |
| `annotations.csv` | Retrieval relevance judgments | 6 | source URL and checksum |
| PEARL-small | Default local phrase embedding model | 3 | Hugging Face revision and model checksum |

Every acquired resource must be recorded in a run manifest before it is used to
produce tracked or generated artifacts.

