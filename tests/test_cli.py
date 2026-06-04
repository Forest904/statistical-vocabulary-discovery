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
