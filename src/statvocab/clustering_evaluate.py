"""Evaluation utilities for Milestone 4 measure clustering."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from statvocab.cluster_measures import CONTROLLED_DOMAINS, CROSS_DOMAIN
from statvocab.config import AppConfig


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _latest_summary(config: AppConfig) -> dict[str, Any] | None:
    candidates = sorted(
        (config.paths.outputs_dir / "clustering").glob("*/clustering_summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return None
    return cast(dict[str, Any], json.loads(candidates[-1].read_text(encoding="utf-8")))


def _manual_review_status(config: AppConfig) -> dict[str, Any]:
    candidates = sorted(
        (config.paths.outputs_dir / "clustering").glob("*/manual_cluster_review_sample.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return {"status": "pending", "message": "Manual cluster review sample is not available."}
    rows = _read_csv(candidates[-1])
    completed = [
        row
        for row in rows
        if row.get("coherence_score")
        or row.get("domain_label_quality")
        or row.get("representative_quality")
    ]
    payload: dict[str, Any] = {
        "status": "available" if completed else "pending",
        "review_sample": str(candidates[-1]),
        "sample_count": len(rows),
        "completed_count": len(completed),
    }
    if not completed:
        return payload

    coherence_values = [
        int(row["coherence_score"])
        for row in completed
        if row.get("coherence_score", "").strip() in {"0", "1", "2"}
    ]
    domain_values = [
        row.get("domain_label_quality", "").strip().casefold()
        for row in completed
        if row.get("domain_label_quality", "").strip()
    ]
    representative_values = [
        row.get("representative_quality", "").strip().casefold()
        for row in completed
        if row.get("representative_quality", "").strip()
    ]
    notes = [
        row.get("notes", "").strip().casefold()
        for row in completed
        if row.get("notes", "").strip()
    ]
    payload.update(
        {
            "mean_coherence": (
                sum(coherence_values) / len(coherence_values) if coherence_values else None
            ),
            "coherent_fraction": (
                sum(1 for value in coherence_values if value >= 1) / len(coherence_values)
                if coherence_values
                else None
            ),
            "strongly_coherent_fraction": (
                sum(1 for value in coherence_values if value == 2) / len(coherence_values)
                if coherence_values
                else None
            ),
            "domain_label_quality_distribution": dict(sorted(Counter(domain_values).items())),
            "domain_label_accuracy": (
                sum(1 for value in domain_values if value == "correct") / len(domain_values)
                if domain_values
                else None
            ),
            "representative_quality_distribution": dict(
                sorted(Counter(representative_values).items())
            ),
            "representative_good_fraction": (
                sum(1 for value in representative_values if value == "good")
                / len(representative_values)
                if representative_values
                else None
            ),
            "note_distribution": dict(sorted(Counter(notes).items())),
        }
    )
    return payload


def evaluate_clustering(
    config: AppConfig,
    *,
    output_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Evaluate clustering artifacts and manual review status."""

    cluster_path = config.paths.outputs_dir / "measure_clusters.csv"
    rows = _read_csv(cluster_path)
    allowed_domains = set(CONTROLLED_DOMAINS)
    failures: list[str] = []
    term_ids = [row.get("term_id", "") for row in rows]
    duplicate_ids = sorted(term_id for term_id, count in Counter(term_ids).items() if count > 1)
    invalid_domains = sorted({row.get("domain", "") for row in rows} - allowed_domains)
    measure_rows = _read_csv(config.paths.outputs_dir / "measures.csv")
    valid_measure_ids = {row.get("term_id", "") for row in measure_rows}
    if duplicate_ids:
        failures.append(f"duplicate clustered measure IDs: {', '.join(duplicate_ids[:10])}")
    if invalid_domains:
        failures.append(f"invalid domains: {', '.join(invalid_domains)}")
    if valid_measure_ids:
        unknown_measure_ids = sorted(set(term_ids) - valid_measure_ids)
        missing_measure_ids = sorted(valid_measure_ids - set(term_ids))
        if unknown_measure_ids:
            failures.append(
                "cluster rows reference unknown measures: " + ", ".join(unknown_measure_ids[:10])
            )
        if missing_measure_ids:
            failures.append(
                "final measures missing from clustering: " + ", ".join(missing_measure_ids[:10])
            )

    non_noise = [row for row in rows if row.get("cluster_id") != "unclustered"]
    representative_rows = [row for row in rows if row.get("is_representative") == "true"]
    invalid_representatives = [
        row["term_id"]
        for row in representative_rows
        if row.get("cluster_id") == "unclustered"
    ]
    if invalid_representatives:
        failures.append(
            "unclustered rows marked representative: " + ", ".join(invalid_representatives[:10])
        )

    domain_counts = Counter(row.get("domain", "") for row in rows)
    cluster_counts = Counter(row.get("cluster_id", "") for row in rows)
    summary = _latest_summary(config)
    payload: dict[str, Any] = {
        "area": "clustering",
        "config_name": config.config_name,
        "corpus": config.corpus.name,
        "artifact": str(cluster_path),
        "artifact_status": "available" if rows else "pending",
        "measure_count": len(rows),
        "cluster_count": len({row.get("cluster_id") for row in non_noise}),
        "unclustered_count": len(rows) - len(non_noise),
        "coverage": len(non_noise) / len(rows) if rows else 0.0,
        "cluster_size_distribution": dict(sorted(cluster_counts.items())),
        "domain_distribution": dict(sorted(domain_counts.items())),
        "cross_domain_or_other_count": domain_counts.get(CROSS_DOMAIN, 0),
        "representative_count": len(representative_rows),
        "manual_review": _manual_review_status(config),
        "latest_run_summary": summary or {},
        "validation": {"passed": not failures, "failures": failures},
    }
    if summary:
        payload["silhouette"] = summary.get("silhouette")
        payload["hdbscan_stability_proxy"] = summary.get("hdbscan_stability_proxy")
        payload["baseline"] = summary.get("baseline", {})
    else:
        payload["metrics_status"] = "pending_clusters"

    metrics_path = output_path or config.paths.reports_dir / "clustering_metrics.json"
    return _write_json(payload, metrics_path), payload
