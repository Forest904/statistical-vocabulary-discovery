import json

from statvocab.benchmark_core import write_core_benchmark
from statvocab.config import load_config


def test_core_benchmark_report_contains_required_fields(tmp_path) -> None:
    config = load_config("configs/core.yaml")
    config = config.model_copy(
        update={
            "paths": config.paths.model_copy(
                update={
                    "processed_dir": tmp_path / "processed",
                    "outputs_dir": tmp_path / "outputs",
                    "reports_dir": tmp_path / "report",
                }
            )
        }
    )
    config.paths.processed_dir.mkdir(parents=True)
    config.paths.outputs_dir.mkdir(parents=True)
    config.paths.reports_dir.mkdir(parents=True)
    (config.paths.processed_dir / "extraction_diagnostics.json").write_text(
        json.dumps(
            {
                "completed_table_count": 2000,
                "vocabulary_count": 100,
                "wall_clock_seconds": 10.0,
                "tables_per_minute": 12000.0,
                "parallel_workers": 4,
                "parallel_chunk_size": 25,
            }
        ),
        encoding="utf-8",
    )
    (config.paths.reports_dir / "relations_metrics.json").write_text(
        json.dumps(
            {
                "latest_run_summary": {
                    "candidate_count": 50,
                    "accepted_count": 25,
                    "wall_clock_seconds": 5.0,
                    "relation_candidates_per_second": 10.0,
                    "all_pair_count": 100,
                    "topk_pair_count": 50,
                    "semantic_neighbor_k": 50,
                }
            }
        ),
        encoding="utf-8",
    )
    (config.paths.reports_dir / "retrieval_metrics.json").write_text(
        json.dumps(
            {
                "index": {
                    "run_id": "run_index",
                    "build_seconds": 4.0,
                    "document_count": 2000,
                    "total_index_size_bytes": 123,
                },
                "tuning": {"selected_preset": "prd_default"},
            }
        ),
        encoding="utf-8",
    )

    path, payload = write_core_benchmark(config)

    assert path == config.paths.reports_dir / "performance_metrics.json"
    assert payload["stages"]["extraction"]["wall_clock_seconds"] == 10.0
    assert payload["stages"]["relations"]["diagnostics"]["topk_pair_count"] == 50
    assert payload["selected_config"]["extraction_parallel_workers"] == 4
    assert "artifact_sizes" in payload
