"""Command-line interface for StatVocab."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from typer.models import OptionInfo

from statvocab import __version__
from statvocab.config import load_config
from statvocab.logging import configure_logging

app = typer.Typer(
    help="Grounded vocabulary discovery for Eurostat-style statistical tables.",
    no_args_is_help=True,
)
console = Console()


def _config_option() -> OptionInfo:
    return cast(OptionInfo, typer.Option("--config", "-c", help="Path to a StatVocab YAML config."))


@app.callback()
def main() -> None:
    """StatVocab command group."""

    configure_logging()


@app.command()
def version() -> None:
    """Print the installed StatVocab version."""

    console.print(__version__)


@app.command("config-check")
def config_check(config: Annotated[Path, _config_option()] = Path("configs/core.yaml")) -> None:
    """Validate a StatVocab YAML configuration."""

    loaded = load_config(config)
    console.print(
        {
            "config_name": loaded.config_name,
            "corpus": loaded.corpus.name,
            "expected_table_count": loaded.corpus.expected_table_count,
        }
    )


def _not_ready(command: str, milestone: str) -> None:
    raise typer.BadParameter(
        f"`statvocab {command}` is reserved for {milestone}; "
        "Milestone 0 only defines the CLI contract."
    )


@app.command()
def acquire(config: Annotated[Path, _config_option()] = Path("configs/core.yaml")) -> None:
    """Acquire and verify source resources. Reserved for Milestone 1."""

    _ = config
    _not_ready("acquire", "Milestone 1")


@app.command()
def ingest(config: Annotated[Path, _config_option()] = Path("configs/core.yaml")) -> None:
    """Ingest source tables. Reserved for Milestone 1."""

    _ = config
    _not_ready("ingest", "Milestone 1")


@app.command()
def extract(config: Annotated[Path, _config_option()] = Path("configs/core.yaml")) -> None:
    """Extract time, geography, and vocabulary artifacts. Reserved for Milestone 2."""

    _ = config
    _not_ready("extract", "Milestone 2")


@app.command()
def classify(
    variant: Annotated[
        str,
        typer.Option("--variant", help="Classifier variant to run."),
    ] = "local-hybrid",
    config: Annotated[Path, _config_option()] = Path("configs/core.yaml"),
) -> None:
    """Partition vocabulary into semantic categories. Reserved for Milestone 3."""

    _ = (config, variant)
    _not_ready("classify", "Milestone 3")


@app.command("cluster-measures")
def cluster_measures(config: Annotated[Path, _config_option()] = Path("configs/core.yaml")) -> None:
    """Cluster measures into domains. Reserved for Milestone 4."""

    _ = config
    _not_ready("cluster-measures", "Milestone 4")


@app.command()
def relations(config: Annotated[Path, _config_option()] = Path("configs/core.yaml")) -> None:
    """Generate measure relationship candidates. Reserved for Milestone 5."""

    _ = config
    _not_ready("relations", "Milestone 5")


@app.command("build-search-index")
def build_search_index(
    config: Annotated[Path, _config_option()] = Path("configs/core.yaml"),
) -> None:
    """Build lexical and semantic search indexes. Reserved for Milestone 6."""

    _ = config
    _not_ready("build-search-index", "Milestone 6")


@app.command()
def evaluate(
    area: Annotated[str | None, typer.Option("--area", help="Evaluation area to run.")] = None,
    config: Annotated[Path, _config_option()] = Path("configs/evaluation.yaml"),
) -> None:
    """Run quality evaluations. Reserved for later milestones."""

    _ = (config, area)
    _not_ready("evaluate", "Milestones 2-6")


@app.command("run-all")
def run_all(config: Annotated[Path, _config_option()] = Path("configs/core.yaml")) -> None:
    """Run the full pipeline. Reserved for final reproducibility milestones."""

    _ = config
    _not_ready("run-all", "Milestone 11")


@app.command("validate-artifacts")
def validate_artifacts(
    run_id: Annotated[str, typer.Option("--run-id", help="Run ID to validate.")],
) -> None:
    """Validate generated artifacts. Reserved for later milestones."""

    _ = run_id
    _not_ready("validate-artifacts", "Milestone 3 and later")
