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

