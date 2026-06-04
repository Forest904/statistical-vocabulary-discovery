"""Milestone 4 measure-domain clustering runner."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from statvocab.classification_features import feature_rows, read_parquet_rows, write_csv_rows
from statvocab.classify_embeddings import generate_term_embeddings
from statvocab.config import AppConfig
from statvocab.contracts import ResourceRecord, stable_id
from statvocab.manifests import complete_manifest, create_manifest, write_manifest

CROSS_DOMAIN = "cross-domain or other"

CONTROLLED_DOMAINS: dict[str, str] = {
    "agriculture, forestry, and fisheries": (
        "Agriculture, crops, livestock, fisheries, forestry, land use, food production."
    ),
    "economy and finance": (
        "Prices, national accounts, money, income, expenditure, finance, trade balances."
    ),
    "population and demography": (
        "Population, births, deaths, migration, age, households, demographic change."
    ),
    "labour market": (
        "Employment, unemployment, labour force, jobs, wages, working time, occupations."
    ),
    "education": "Schools, pupils, students, education attainment, training, learning.",
    "health": "Health, mortality, morbidity, care, hospitals, disease, wellbeing.",
    "social conditions and equality": (
        "Poverty, inequality, social protection, living conditions, inclusion, gender equality."
    ),
    "industry, trade, and services": (
        "Industrial production, business, retail, services, tourism, enterprise activity."
    ),
    "transport": "Transport, vehicles, passengers, freight, roads, rail, air, maritime movement.",
    "environment and energy": (
        "Environment, emissions, climate, waste, water, energy production, energy consumption."
    ),
    "science, technology, and digital society": (
        "Research, development, innovation, technology, ICT, digital society."
    ),
    "government, justice, and public administration": (
        "Government, public administration, elections, justice, crime, courts, public finance."
    ),
    "regional and urban statistics": (
        "Regions, cities, urban areas, territorial statistics, local indicators."
    ),
    CROSS_DOMAIN: "Cross-domain, mixed, generic, or insufficiently specific statistical measures.",
}

MEASURE_CLUSTER_FIELDNAMES = [
    "term_id",
    "term",
    "cluster_id",
    "domain",
    "membership_probability",
    "is_representative",
    "labeling_method",
    "evidence",
]
BASELINE_FIELDNAMES = ["term_id", "term", "baseline_cluster_id", "method", "evidence"]
REVIEW_FIELDNAMES = [
    "term_id",
    "term",
    "cluster_id",
    "domain",
    "is_representative",
    "coherence_score",
    "domain_label_quality",
    "representative_quality",
    "notes",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run `statvocab classify` first.")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _import_clustering_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        numpy = import_module("numpy")
        hdbscan = import_module("hdbscan")
        cluster = import_module("sklearn.cluster")
        metrics = import_module("sklearn.metrics")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Measure clustering requires the ML extra, including hdbscan and scikit-learn. "
            "Install with `python -m pip install -e .[ml]`."
        ) from exc
    return numpy, hdbscan, cluster.AgglomerativeClustering, metrics.silhouette_score


def _measure_rows(config: AppConfig) -> list[dict[str, str]]:
    rows = _read_csv(config.paths.outputs_dir / "measures.csv")
    rows.sort(key=lambda row: (row["canonical_term"].casefold(), row["term_id"]))
    return rows


def _ensure_embeddings(config: AppConfig, measures: list[dict[str, str]]) -> dict[str, list[float]]:
    path = config.paths.processed_dir / "term_embeddings.parquet"
    if not path.exists():
        generate_term_embeddings(config, feature_rows(config))
    rows = read_parquet_rows(path)
    by_term = {
        str(row["term_id"]): [float(value) for value in cast(list[Any], row["embedding"])]
        for row in rows
        if row.get("model") == config.classification.pearl_model
        and row.get("revision") == config.classification.pearl_revision
    }
    missing = sorted(row["term_id"] for row in measures if row["term_id"] not in by_term)
    if missing:
        generate_term_embeddings(config, feature_rows(config))
        rows = read_parquet_rows(path)
        by_term = {
            str(row["term_id"]): [float(value) for value in cast(list[Any], row["embedding"])]
            for row in rows
            if row.get("model") == config.classification.pearl_model
            and row.get("revision") == config.classification.pearl_revision
        }
        missing = sorted(row["term_id"] for row in measures if row["term_id"] not in by_term)
    if missing:
        raise RuntimeError(f"Missing PEARL embeddings for {len(missing)} measure terms.")
    return by_term


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    size = len(vectors[0])
    values = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(size)]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(left, right, strict=True)))


def _cluster_id(label: int) -> str:
    return "unclustered" if label < 0 else stable_id("cluster", "hdbscan", label, digest_size=6)


def _baseline_id(label: int) -> str:
    return stable_id("cluster", "agglomerative", label, digest_size=6)


def _domain_embeddings(config: AppConfig, measure_count: int) -> dict[str, list[float]]:
    rows = [
        {
            "term_id": stable_id("term", "domain", domain, digest_size=6),
            "canonical_term": f"{domain}: {description}",
        }
        for domain, description in CONTROLLED_DOMAINS.items()
    ]
    _ = measure_count
    generated = generate_term_embeddings(
        config,
        rows,
        config.paths.processed_dir / "domain_taxonomy_embeddings.parquet",
    )
    embedding_rows = read_parquet_rows(generated)
    return {
        str(row["canonical_term"]).split(":", maxsplit=1)[0]: [
            float(value) for value in cast(list[Any], row["embedding"])
        ]
        for row in embedding_rows
    }


def _label_clusters(
    config: AppConfig,
    cluster_vectors: dict[str, list[list[float]]],
    domain_vectors: dict[str, list[float]],
) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for cluster_id, vectors in cluster_vectors.items():
        if cluster_id == "unclustered":
            labels[cluster_id] = {
                "domain": CROSS_DOMAIN,
                "labeling_method": "noise_default",
                "domain_scores": {},
            }
            continue
        center = _centroid(vectors)
        scores = {
            domain: _dot(center, vector)
            for domain, vector in domain_vectors.items()
            if domain != CROSS_DOMAIN
        }
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_domain, best_score = ordered[0] if ordered else (CROSS_DOMAIN, 0.0)
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        if (
            best_score >= config.clustering.domain_similarity_threshold
            and best_score - second_score >= config.clustering.domain_similarity_margin
        ):
            domain = best_domain
            method = "embedding_threshold"
        else:
            domain = CROSS_DOMAIN
            method = "embedding_threshold_fallback"
        labels[cluster_id] = {
            "domain": domain,
            "labeling_method": method,
            "domain_scores": dict(ordered[:5]),
        }
    return labels


def _representatives(
    rows: list[dict[str, str]],
    labels: list[int],
    vectors_by_term: dict[str, list[float]],
    config: AppConfig,
) -> set[str]:
    representative_ids: set[str] = set()
    for label in sorted({value for value in labels if value >= 0}):
        members = [row for row, row_label in zip(rows, labels, strict=True) if row_label == label]
        center = _centroid([vectors_by_term[row["term_id"]] for row in members])
        ranked = sorted(
            members,
            key=lambda row: (_distance(vectors_by_term[row["term_id"]], center), row["term_id"]),
        )
        representative_ids.update(
            row["term_id"] for row in ranked[: config.clustering.representative_count]
        )
    return representative_ids


def _hdbscan_labels(model: Any) -> list[int]:
    return [int(value) for value in list(model.labels_)]


def _hdbscan_probabilities(model: Any, count: int) -> list[float]:
    values = getattr(model, "probabilities_", None)
    if values is None:
        return [1.0] * count
    return [float(value) for value in list(values)]


def _cluster_vectors(
    rows: list[dict[str, str]],
    labels: list[int],
    vectors_by_term: dict[str, list[float]],
) -> dict[str, list[list[float]]]:
    grouped: dict[str, list[list[float]]] = {}
    for row, label in zip(rows, labels, strict=True):
        grouped.setdefault(_cluster_id(label), []).append(vectors_by_term[row["term_id"]])
    return grouped


def _silhouette(
    silhouette_score: Any,
    matrix: Any,
    labels: list[int],
) -> float | None:
    clustered = [label for label in labels if label >= 0]
    if len(set(clustered)) < 2 or len(clustered) != len(labels):
        return None
    try:
        return float(silhouette_score(matrix, labels, metric="euclidean"))
    except ValueError:
        return None


def _cluster_rows(
    rows: list[dict[str, str]],
    labels: list[int],
    probabilities: list[float],
    representative_ids: set[str],
    cluster_domains: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, label, probability in zip(rows, labels, probabilities, strict=True):
        cluster_id = _cluster_id(label)
        domain_info = cluster_domains[cluster_id]
        output.append(
            {
                "term_id": row["term_id"],
                "term": row["canonical_term"],
                "cluster_id": cluster_id,
                "domain": domain_info["domain"],
                "membership_probability": f"{probability:.6f}",
                "is_representative": str(row["term_id"] in representative_ids).lower(),
                "labeling_method": domain_info["labeling_method"],
                "evidence": json.dumps(
                    {
                        "source": "hdbscan_pearl_small",
                        "classification_run_id": row.get("run_id", ""),
                        "domain_scores": domain_info.get("domain_scores", {}),
                    },
                    sort_keys=True,
                ),
            }
        )
    return output


def _baseline_rows(
    rows: list[dict[str, str]],
    baseline_labels: list[int],
    config: AppConfig,
) -> list[dict[str, str]]:
    return [
        {
            "term_id": row["term_id"],
            "term": row["canonical_term"],
            "baseline_cluster_id": _baseline_id(label),
            "method": "agglomerative",
            "evidence": json.dumps(
                {
                    "distance_threshold": config.clustering.agglomerative_distance_threshold,
                    "linkage": config.clustering.agglomerative_linkage,
                },
                sort_keys=True,
            ),
        }
        for row, label in zip(rows, baseline_labels, strict=True)
    ]


def _review_rows(rows: list[dict[str, Any]], config: AppConfig) -> list[dict[str, Any]]:
    sample_size = min(config.clustering.manual_review_sample_size, len(rows))
    rng = random.Random(config.random_seed)
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cluster.setdefault(str(row["cluster_id"]), []).append(row)
    selected: dict[str, dict[str, Any]] = {}
    for cluster_id in sorted(by_cluster):
        candidates = sorted(by_cluster[cluster_id], key=lambda row: (row["term"], row["term_id"]))
        selected[str(candidates[0]["term_id"])] = candidates[0]
    remaining = [row for row in rows if str(row["term_id"]) not in selected]
    remaining.sort(key=lambda row: str(row["term_id"]))
    if len(selected) < sample_size and remaining:
        selected.update(
            {
                str(row["term_id"]): row
                for row in rng.sample(remaining, k=min(sample_size - len(selected), len(remaining)))
            }
        )
    review_rows = []
    for row in sorted(
        selected.values(),
        key=lambda item: (str(item["cluster_id"]), str(item["term"])),
    ):
        review_rows.append(
            {
                "term_id": row["term_id"],
                "term": row["term"],
                "cluster_id": row["cluster_id"],
                "domain": row["domain"],
                "is_representative": row["is_representative"],
                "coherence_score": "",
                "domain_label_quality": "",
                "representative_quality": "",
                "notes": "",
            }
        )
    return review_rows


def _summary(
    rows: list[dict[str, Any]],
    labels: list[int],
    baseline_labels: list[int],
    probabilities: list[float],
    silhouette: float | None,
) -> dict[str, Any]:
    cluster_counts = Counter(str(row["cluster_id"]) for row in rows)
    domain_counts = Counter(str(row["domain"]) for row in rows)
    non_noise = sum(1 for label in labels if label >= 0)
    persistence_values = [
        float(value)
        for value in probabilities
        if not math.isnan(float(value))
    ]
    return {
        "measure_count": len(rows),
        "cluster_count": len({label for label in labels if label >= 0}),
        "unclustered_count": len(rows) - non_noise,
        "coverage": non_noise / len(rows) if rows else 0.0,
        "cluster_size_distribution": dict(sorted(cluster_counts.items())),
        "domain_distribution": dict(sorted(domain_counts.items())),
        "silhouette": silhouette,
        "hdbscan_stability_proxy": (
            sum(persistence_values) / len(persistence_values) if persistence_values else None
        ),
        "baseline": {
            "method": "agglomerative",
            "cluster_count": len(set(baseline_labels)),
            "distance_threshold": "configured",
        },
    }


def run_measure_clustering(
    config: AppConfig,
    *,
    resources: tuple[ResourceRecord, ...] = (),
) -> tuple[dict[str, Path], Path, dict[str, Any]]:
    """Cluster final measure terms and export domain review artifacts."""

    numpy, hdbscan, agglomerative_clustering, silhouette_score = _import_clustering_dependencies()
    measures = _measure_rows(config)
    manifest = create_manifest(config, "cluster-measures")
    output_dir = config.paths.outputs_dir / "clustering" / manifest.run_id

    if not measures:
        rows: list[dict[str, Any]] = []
        summary = {"run_id": manifest.run_id, "measure_count": 0, "coverage": 0.0}
        artifacts = {
            "measure_clusters": write_csv_rows(
                rows,
                config.paths.outputs_dir / "measure_clusters.csv",
                MEASURE_CLUSTER_FIELDNAMES,
            ),
            "baseline_assignments": write_csv_rows(
                rows,
                output_dir / "agglomerative_baseline.csv",
                BASELINE_FIELDNAMES,
            ),
            "domain_taxonomy": _write_json(CONTROLLED_DOMAINS, output_dir / "domain_taxonomy.json"),
            "manual_review_sample": write_csv_rows(
                rows,
                output_dir / "manual_cluster_review_sample.csv",
                REVIEW_FIELDNAMES,
            ),
            "summary": _write_json(
                summary,
                output_dir / "clustering_summary.json",
            ),
        }
        completed = complete_manifest(
            manifest,
            resources=tuple(record.resource_id for record in resources),
            artifacts=tuple(str(path) for path in artifacts.values()),
        )
        manifest_path = write_manifest(
            completed,
            config.paths.outputs_dir / "manifests" / f"{manifest.run_id}_cluster_measures.json",
        )
        return artifacts, manifest_path, summary

    vectors_by_term = _ensure_embeddings(config, measures)
    matrix = numpy.array([vectors_by_term[row["term_id"]] for row in measures], dtype=float)
    min_cluster_size = min(config.clustering.hdbscan_min_cluster_size, len(measures))
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min(config.clustering.hdbscan_min_samples, min_cluster_size),
        metric=config.clustering.hdbscan_metric,
        prediction_data=False,
    )
    model.fit(matrix)
    labels = _hdbscan_labels(model)
    probabilities = _hdbscan_probabilities(model, len(measures))

    baseline = agglomerative_clustering(
        n_clusters=None,
        distance_threshold=config.clustering.agglomerative_distance_threshold,
        metric="cosine",
        linkage=config.clustering.agglomerative_linkage,
    )
    baseline.fit(matrix)
    baseline_labels = [int(value) for value in list(baseline.labels_)]

    representative_ids = _representatives(measures, labels, vectors_by_term, config)
    domain_vectors = _domain_embeddings(config, len(measures))
    cluster_domains = _label_clusters(
        config,
        _cluster_vectors(measures, labels, vectors_by_term),
        domain_vectors,
    )
    rows = _cluster_rows(measures, labels, probabilities, representative_ids, cluster_domains)
    baseline_export_rows = _baseline_rows(measures, baseline_labels, config)
    review_rows = _review_rows(rows, config)
    summary = _summary(
        rows,
        labels,
        baseline_labels,
        probabilities,
        _silhouette(silhouette_score, matrix, labels),
    )
    summary["run_id"] = manifest.run_id
    summary["labeling"] = {
        "controlled_domain_count": len(CONTROLLED_DOMAINS),
        "similarity_threshold": config.clustering.domain_similarity_threshold,
        "similarity_margin": config.clustering.domain_similarity_margin,
    }

    artifacts = {
        "measure_clusters": write_csv_rows(
            rows,
            config.paths.outputs_dir / "measure_clusters.csv",
            MEASURE_CLUSTER_FIELDNAMES,
        ),
        "baseline_assignments": write_csv_rows(
            baseline_export_rows,
            output_dir / "agglomerative_baseline.csv",
            BASELINE_FIELDNAMES,
        ),
        "domain_taxonomy": _write_json(CONTROLLED_DOMAINS, output_dir / "domain_taxonomy.json"),
        "manual_review_sample": write_csv_rows(
            review_rows,
            output_dir / "manual_cluster_review_sample.csv",
            REVIEW_FIELDNAMES,
        ),
        "summary": _write_json(summary, output_dir / "clustering_summary.json"),
    }
    completed = complete_manifest(
        manifest,
        resources=tuple(record.resource_id for record in resources),
        artifacts=tuple(str(path) for path in artifacts.values()),
    )
    manifest_path = write_manifest(
        completed,
        config.paths.outputs_dir / "manifests" / f"{manifest.run_id}_cluster_measures.json",
    )
    return artifacts, manifest_path, summary
