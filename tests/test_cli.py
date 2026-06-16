from typer.testing import CliRunner

from statvocab.cli import app


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Grounded vocabulary discovery" in result.output


def test_config_check() -> None:
    result = CliRunner().invoke(app, ["config-check", "--config", "configs/core.yaml"])

    assert result.exit_code == 0
    assert "core" in result.output
    assert "2000" in result.output


def test_cluster_measures_cli_invokes_runner(monkeypatch) -> None:
    from pathlib import Path

    import statvocab.cli as cli

    def fake_run(config, *, resources=()):
        _ = config, resources
        return (
            {"measure_clusters": Path("outputs/measure_clusters.csv")},
            Path("outputs/manifests/run_cluster_measures.json"),
            {"run_id": "run_test", "measure_count": 1, "coverage": 1.0, "cluster_count": 1},
        )

    monkeypatch.setattr(cli, "run_measure_clustering", fake_run)

    result = CliRunner().invoke(app, ["cluster-measures", "--config", "configs/evaluation.yaml"])

    assert result.exit_code == 0
    assert "run_test" in result.output


def test_clustering_evaluate_cli_invokes_evaluator(monkeypatch) -> None:
    from pathlib import Path

    import statvocab.cli as cli

    def fake_evaluate(config):
        _ = config
        return (
            Path("report/clustering_metrics.json"),
            {
                "artifact_status": "available",
                "coverage": 1.0,
                "validation": {"passed": True},
            },
        )

    monkeypatch.setattr(cli, "evaluate_clustering", fake_evaluate)

    result = CliRunner().invoke(
        app,
        ["evaluate", "--area", "clustering", "--config", "configs/evaluation.yaml"],
    )

    assert result.exit_code == 0
    assert "clustering" in result.output


def test_classification_evaluate_cli_reports_export_source(monkeypatch) -> None:
    from pathlib import Path

    import statvocab.cli as cli

    def fake_evaluate(config):
        _ = config
        return (
            Path("report/classification_metrics.json"),
            {
                "gold_status": "available",
                "evaluated_prediction_source": "current_category_exports",
            },
        )

    monkeypatch.setattr(cli, "evaluate_classification", fake_evaluate)

    result = CliRunner().invoke(
        app,
        ["evaluate", "--area", "classification", "--config", "configs/evaluation.yaml"],
    )

    assert result.exit_code == 0
    assert "available" in result.output
    assert "current_category_exports" in result.output


def test_relations_cli_invokes_runner(monkeypatch) -> None:
    from pathlib import Path

    import statvocab.cli as cli

    def fake_run(config, *, resources=()):
        _ = config, resources
        return (
            {"measure_relations": Path("outputs/measure_relations.csv")},
            Path("outputs/manifests/run_relations.json"),
            {"run_id": "run_test", "candidate_count": 2, "accepted_count": 1},
        )

    monkeypatch.setattr(cli, "run_measure_relations", fake_run)

    result = CliRunner().invoke(app, ["relations", "--config", "configs/evaluation.yaml"])

    assert result.exit_code == 0
    assert "run_test" in result.output


def test_relations_evaluate_cli_invokes_evaluator(monkeypatch) -> None:
    from pathlib import Path

    import statvocab.cli as cli

    def fake_evaluate(config):
        _ = config
        return (
            Path("report/relations_metrics.json"),
            {
                "artifact_status": "available",
                "relation_count": 1,
                "validation": {"passed": True},
            },
        )

    monkeypatch.setattr(cli, "evaluate_relations", fake_evaluate)

    result = CliRunner().invoke(
        app,
        ["evaluate", "--area", "relations", "--config", "configs/evaluation.yaml"],
    )

    assert result.exit_code == 0
    assert "relations" in result.output


def test_build_search_index_cli_invokes_runner(monkeypatch) -> None:
    from pathlib import Path

    import statvocab.cli as cli

    def fake_run(config, *, resources=()):
        _ = config, resources
        return (
            {"lexical_index": Path("outputs/search/run_test/lexical.sqlite")},
            Path("outputs/manifests/run_search.json"),
            {
                "run_id": "run_test",
                "document_count": 4,
                "embedding_dimension": 768,
                "total_index_size_bytes": 123,
            },
        )

    monkeypatch.setattr(cli, "run_search_index_build", fake_run)

    result = CliRunner().invoke(app, ["build-search-index", "--config", "configs/evaluation.yaml"])

    assert result.exit_code == 0
    assert "run_test" in result.output


def test_retrieval_evaluate_cli_invokes_evaluator(monkeypatch) -> None:
    from pathlib import Path

    import statvocab.cli as cli

    def fake_evaluate(config):
        _ = config
        return (
            Path("report/retrieval_metrics.json"),
            {
                "tuning": {"selected_preset": "prd_default"},
                "index": {"run_id": "run_index"},
            },
        )

    monkeypatch.setattr(cli, "evaluate_retrieval", fake_evaluate)

    result = CliRunner().invoke(
        app,
        ["evaluate", "--area", "retrieval", "--config", "configs/evaluation.yaml"],
    )

    assert result.exit_code == 0
    assert "retrieval" in result.output
    assert "prd_default" in result.output


def test_run_all_cli_invokes_full_corpus_runner(monkeypatch) -> None:
    import statvocab.cli as cli

    def fake_run(config, *, run_id=None, resume=False):
        _ = run_id, resume
        return {
            "run_id": "run_full_test",
            "run_manifest": str(config.paths.outputs_dir / "full_corpus" / "run_manifest.json"),
            "stage_status_counts": {"succeeded": 1, "failed": 0, "blocked": 0, "skipped": 0},
        }

    monkeypatch.setattr(cli, "run_full_corpus", fake_run)

    result = CliRunner().invoke(app, ["run-all", "--config", "configs/full.yaml"])

    assert result.exit_code == 0
    assert "run_full_test" in result.output


def test_run_all_cli_audit_resume_does_not_run(monkeypatch) -> None:
    import statvocab.cli as cli

    def fake_audit(config, *, run_id):
        return {"run_id": run_id, "config": config.config_name, "stages": []}

    def fail_run(*_args, **_kwargs):
        raise AssertionError("audit mode must not execute run_full_corpus")

    monkeypatch.setattr(cli, "audit_full_corpus_resume", fake_audit)
    monkeypatch.setattr(cli, "run_full_corpus", fail_run)

    result = CliRunner().invoke(
        app,
        [
            "run-all",
            "--config",
            "configs/full.yaml",
            "--resume-run-id",
            "run_test",
            "--audit-resume",
        ],
    )

    assert result.exit_code == 0
    assert "run_test" in result.output


def test_core_release_cli_commands(monkeypatch) -> None:
    from pathlib import Path

    import statvocab.cli as cli

    monkeypatch.setattr(
        cli,
        "write_core_release_manifest",
        lambda config: Path("report/core_release_manifest.json"),
    )
    monkeypatch.setattr(
        cli,
        "validate_core_release_manifest",
        lambda config: {"validated": True, "file_count": 1},
    )

    freeze = CliRunner().invoke(app, ["freeze-core-release", "--config", "configs/core.yaml"])
    validate = CliRunner().invoke(app, ["validate-core-release", "--config", "configs/core.yaml"])

    assert freeze.exit_code == 0
    assert "core_release_manifest" in freeze.output
    assert validate.exit_code == 0
    assert "validated" in validate.output
