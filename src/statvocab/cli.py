"""Command-line interface for StatVocab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from typer.models import OptionInfo

from statvocab import __version__
from statvocab.artifact_validation import (
    validate_artifacts as run_artifact_validation,
)
from statvocab.artifact_validation import (
    validate_core_release_manifest,
    write_core_release_manifest,
)
from statvocab.benchmark_core import write_core_benchmark
from statvocab.classification import run_classification
from statvocab.classification_evaluate import evaluate_classification
from statvocab.cluster_measures import run_measure_clustering
from statvocab.clustering_evaluate import evaluate_clustering
from statvocab.config import load_config
from statvocab.contracts import ResourceRecord
from statvocab.extraction_evaluate import run_extraction_evaluation
from statvocab.full_corpus import audit_full_corpus_resume, run_full_corpus
from statvocab.ingest import run_ingestion
from statvocab.knowledge_graph import run_knowledge_graph
from statvocab.logging import configure_logging
from statvocab.manifests import complete_manifest, create_manifest, write_manifest
from statvocab.relations import run_measure_relations
from statvocab.relations_evaluate import evaluate_relations
from statvocab.resources import acquire_resources, count_csv_files, write_resource_manifest
from statvocab.retrieval_evaluate import evaluate_retrieval
from statvocab.search import build_search_index as run_search_index_build
from statvocab.targeted_review import ensure_targeted_reclaim_templates
from statvocab.vocabulary import run_extraction

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
    """Acquire and verify source resources."""

    loaded = load_config(config)
    manifest = create_manifest(loaded, "acquire")
    records = acquire_resources(loaded)
    resource_manifest_path = write_resource_manifest(
        records,
        loaded.paths.processed_dir / "resource_manifest.json",
    )
    completed = complete_manifest(
        manifest,
        resources=tuple(record.resource_id for record in records),
        artifacts=(str(resource_manifest_path),),
    )
    run_manifest_path = write_manifest(
        completed,
        loaded.paths.outputs_dir / "manifests" / f"{manifest.run_id}_acquire.json",
    )
    archive_name = loaded.corpus.archive_name or ""
    extracted_count = count_csv_files(loaded.paths.raw_dir / archive_name.removesuffix(".tgz"))
    console.print(
        {
            "resource_manifest": str(resource_manifest_path),
            "run_manifest": str(run_manifest_path),
            "resources": {record.name: record.validation_status.value for record in records},
            "extracted_table_count": extracted_count,
        }
    )


@app.command()
def ingest(config: Annotated[Path, _config_option()] = Path("configs/core.yaml")) -> None:
    """Ingest source tables into the Milestone 1 table inventory."""

    loaded = load_config(config)
    resources = _load_resource_records(loaded.paths.processed_dir / "resource_manifest.json")
    tables_path, diagnostics_path, manifest_path, diagnostics = run_ingestion(
        loaded,
        resources=resources,
    )
    console.print(
        {
            "tables": str(tables_path),
            "diagnostics": str(diagnostics_path),
            "run_manifest": str(manifest_path),
            "inventoried_table_count": diagnostics["inventoried_table_count"],
            "status_counts": diagnostics["status_counts"],
        }
    )


def _load_resource_records(path: Path) -> tuple[ResourceRecord, ...]:
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(ResourceRecord.model_validate(item) for item in payload)


@app.command()
def extract(config: Annotated[Path, _config_option()] = Path("configs/core.yaml")) -> None:
    """Extract time, geography, title terms, and global vocabulary artifacts."""

    loaded = load_config(config)
    resources = _load_resource_records(loaded.paths.processed_dir / "resource_manifest.json")
    artifacts, manifest_path, diagnostics = run_extraction(loaded, resources=resources)
    console.print(
        {
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "run_manifest": str(manifest_path),
            "table_count": diagnostics["table_count"],
            "vocabulary_count": diagnostics["vocabulary_count"],
            "term_occurrence_count": diagnostics["term_occurrence_count"],
            "geography_variant_for_vocabulary": diagnostics["geography_variant_for_vocabulary"],
        }
    )


@app.command()
def classify(
    variant: Annotated[
        str,
        typer.Option("--variant", help="Classifier variant to run."),
    ] = "local-hybrid",
    config: Annotated[Path, _config_option()] = Path("configs/core.yaml"),
) -> None:
    """Partition vocabulary into semantic categories."""

    loaded = load_config(config)
    resources = _load_resource_records(loaded.paths.processed_dir / "resource_manifest.json")
    artifacts, manifest_path, diagnostics = run_classification(
        loaded,
        variant=variant,
        resources=resources,
    )
    console.print(
        {
            "variant": variant,
            "run_id": diagnostics["run_id"],
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "run_manifest": str(manifest_path),
            "category_counts": diagnostics["category_counts"],
            "metrics_status": diagnostics["metrics_status"],
        }
    )


@app.command("prepare-targeted-review")
def prepare_targeted_review(
    config: Annotated[Path, _config_option()] = Path("configs/core.yaml"),
) -> None:
    """Create targeted reclaim review templates for `other_ambiguous` terms."""

    loaded = load_config(config)
    sample_path, labels_path, relabel_path = ensure_targeted_reclaim_templates(loaded)
    console.print(
        {
            "sample": str(sample_path),
            "labels": str(labels_path),
            "relabel": str(relabel_path),
        }
    )


@app.command("cluster-measures")
def cluster_measures(config: Annotated[Path, _config_option()] = Path("configs/core.yaml")) -> None:
    """Cluster final measures into controlled statistical domains."""

    loaded = load_config(config)
    resources = _load_resource_records(loaded.paths.processed_dir / "resource_manifest.json")
    artifacts, manifest_path, diagnostics = run_measure_clustering(loaded, resources=resources)
    console.print(
        {
            "run_id": diagnostics["run_id"],
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "run_manifest": str(manifest_path),
            "measure_count": diagnostics["measure_count"],
            "coverage": diagnostics.get("coverage", 0.0),
            "cluster_count": diagnostics.get("cluster_count", 0),
        }
    )


@app.command()
def relations(config: Annotated[Path, _config_option()] = Path("configs/core.yaml")) -> None:
    """Generate grounded measure relationship candidates."""

    loaded = load_config(config)
    resources = _load_resource_records(loaded.paths.processed_dir / "resource_manifest.json")
    artifacts, manifest_path, diagnostics = run_measure_relations(loaded, resources=resources)
    console.print(
        {
            "run_id": diagnostics["run_id"],
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "run_manifest": str(manifest_path),
            "candidate_count": diagnostics["candidate_count"],
            "accepted_count": diagnostics["accepted_count"],
        }
    )


@app.command("build-knowledge-graph")
def build_knowledge_graph(
    config: Annotated[Path, _config_option()] = Path("configs/core.yaml"),
) -> None:
    """Build materialized knowledge graph nodes and edges."""

    loaded = load_config(config)
    resources = _load_resource_records(loaded.paths.processed_dir / "resource_manifest.json")
    artifacts, manifest_path, diagnostics = run_knowledge_graph(loaded, resources=resources)
    console.print(
        {
            "run_id": diagnostics["run_id"],
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "run_manifest": str(manifest_path),
            "node_count": diagnostics["node_count"],
            "edge_count": diagnostics["edge_count"],
            "node_type_counts": diagnostics["node_type_counts"],
            "edge_type_counts": diagnostics["edge_type_counts"],
        }
    )


@app.command("build-search-index")
def build_search_index(
    config: Annotated[Path, _config_option()] = Path("configs/core.yaml"),
) -> None:
    """Build lexical and semantic search indexes."""

    loaded = load_config(config)
    resources = _load_resource_records(loaded.paths.processed_dir / "resource_manifest.json")
    artifacts, manifest_path, diagnostics = run_search_index_build(loaded, resources=resources)
    console.print(
        {
            "run_id": diagnostics["run_id"],
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "run_manifest": str(manifest_path),
            "document_count": diagnostics["document_count"],
            "embedding_dimension": diagnostics["embedding_dimension"],
            "total_index_size_bytes": diagnostics["total_index_size_bytes"],
        }
    )


@app.command("benchmark-core")
def benchmark_core(config: Annotated[Path, _config_option()] = Path("configs/core.yaml")) -> None:
    """Write current core-corpus performance diagnostics."""

    loaded = load_config(config)
    metrics_path, payload = write_core_benchmark(loaded)
    console.print(
        {
            "metrics": str(metrics_path),
            "corpus": payload["corpus"],
            "expected_table_count": payload["expected_table_count"],
            "stage_count": len(payload["stages"]),
        }
    )


@app.command()
def evaluate(
    area: Annotated[str | None, typer.Option("--area", help="Evaluation area to run.")] = None,
    config: Annotated[Path, _config_option()] = Path("configs/evaluation.yaml"),
) -> None:
    """Run quality evaluations."""

    loaded = load_config(config)
    if area == "extraction":
        review_path, metrics_path, payload = run_extraction_evaluation(loaded)
        console.print(
            {
                "area": "extraction",
                "review_sample": str(review_path),
                "metrics": str(metrics_path),
                "gold_status": payload["gold_status"],
            }
        )
        return
    if area == "classification":
        metrics_path, payload = evaluate_classification(loaded)
        console.print(
            {
                "area": "classification",
                "metrics": str(metrics_path),
                "gold_status": payload["gold_status"],
                "metrics_status": payload.get("metrics_status", "available"),
                "evaluated_prediction_source": payload.get("evaluated_prediction_source", ""),
            }
        )
        return
    if area == "clustering":
        metrics_path, payload = evaluate_clustering(loaded)
        console.print(
            {
                "area": "clustering",
                "metrics": str(metrics_path),
                "artifact_status": payload["artifact_status"],
                "coverage": payload["coverage"],
                "validation_passed": payload["validation"]["passed"],
            }
        )
        return
    if area == "relations":
        metrics_path, payload = evaluate_relations(loaded)
        console.print(
            {
                "area": "relations",
                "metrics": str(metrics_path),
                "artifact_status": payload["artifact_status"],
                "relation_count": payload["relation_count"],
                "validation_passed": payload["validation"]["passed"],
            }
        )
        return
    if area == "retrieval":
        metrics_path, payload = evaluate_retrieval(loaded)
        console.print(
            {
                "area": "retrieval",
                "metrics": str(metrics_path),
                "selected_preset": payload["tuning"]["selected_preset"],
                "index_run_id": payload.get("index", {}).get("run_id", ""),
            }
        )
        return
    _not_ready("evaluate", "Milestones 3-6 for areas other than extraction")


@app.command("run-all")
def run_all(
    config: Annotated[Path, _config_option()] = Path("configs/full.yaml"),
    resume_run_id: Annotated[
        str | None,
        typer.Option("--resume-run-id", help="Resume a checkpointed full-corpus run ID."),
    ] = None,
    audit_resume: Annotated[
        bool,
        typer.Option("--audit-resume", help="Inspect checkpoint reuse without executing stages."),
    ] = False,
) -> None:
    """Run the instrumented full-corpus scale attempt."""

    loaded = load_config(config)
    if audit_resume:
        if resume_run_id is None:
            raise typer.BadParameter("--audit-resume requires --resume-run-id")
        console.print(audit_full_corpus_resume(loaded, run_id=resume_run_id))
        return
    payload = run_full_corpus(
        loaded,
        run_id=resume_run_id,
        resume=resume_run_id is not None,
    )
    console.print(payload)


@app.command("validate-artifacts")
def validate_artifacts(
    run_id: Annotated[str, typer.Option("--run-id", help="Run ID to validate.")],
    config: Annotated[Path, _config_option()] = Path("configs/core.yaml"),
) -> None:
    """Validate generated artifacts."""

    loaded = load_config(config)
    payload = run_artifact_validation(loaded, run_id=run_id)
    console.print(payload)


@app.command("freeze-core-release")
def freeze_core_release(
    config: Annotated[Path, _config_option()] = Path("configs/core.yaml"),
) -> None:
    """Write a checksum manifest for the current core release artifacts."""

    loaded = load_config(config)
    path = write_core_release_manifest(loaded)
    console.print({"core_release_manifest": str(path)})


@app.command("validate-core-release")
def validate_core_release(
    config: Annotated[Path, _config_option()] = Path("configs/core.yaml"),
) -> None:
    """Validate the frozen core release artifact manifest."""

    loaded = load_config(config)
    console.print(validate_core_release_manifest(loaded))
