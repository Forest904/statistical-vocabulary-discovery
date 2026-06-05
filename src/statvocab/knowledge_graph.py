"""Artifact-backed knowledge graph construction for StatVocab."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path
from typing import Any

from statvocab.classification import CSV_BY_CATEGORY
from statvocab.classification_features import read_parquet_rows, write_parquet_rows
from statvocab.config import AppConfig
from statvocab.contracts import ResourceRecord, stable_id
from statvocab.manifests import complete_manifest, create_manifest, write_manifest
from statvocab.resources import file_md5

NODE_FIELDNAMES = [
    "node_id",
    "node_type",
    "label",
    "properties_json",
    "run_id",
]
EDGE_FIELDNAMES = [
    "edge_id",
    "source_id",
    "target_id",
    "edge_type",
    "weight",
    "directed",
    "derived",
    "evidence_ids_json",
    "properties_json",
    "run_id",
]


def _read_csv(path: Path, *, required: bool = False) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing {path}; run the previous pipeline stage first.")
        return []
    _raise_csv_field_limit()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_list(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _json_dict(value: object) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _node(
    *,
    node_id: str,
    node_type: str,
    label: str,
    properties: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "node_type": node_type,
        "label": label,
        "properties_json": _json(properties),
        "run_id": run_id,
    }


def _edge(
    *,
    edge_id: str,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
    directed: bool,
    derived: bool,
    evidence_ids: Iterable[str] = (),
    properties: dict[str, Any] | None = None,
    run_id: str,
) -> dict[str, Any]:
    return {
        "edge_id": edge_id,
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": edge_type,
        "weight": round(max(0.0, min(1.0, weight)), 6),
        "directed": directed,
        "derived": derived,
        "evidence_ids_json": _json(sorted(set(evidence_ids))),
        "properties_json": _json(properties or {}),
        "run_id": run_id,
    }


def _category_node_id(category: str) -> str:
    return f"category:{category}"


def _domain_node_id(domain: str) -> str:
    return stable_id("graph", "domain", domain, digest_size=8)


def _load_term_outputs(config: AppConfig) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for category, filename in CSV_BY_CATEGORY.items():
        for row in _read_csv(config.paths.outputs_dir / filename, required=True):
            term_id = row.get("term_id", "")
            if not term_id:
                continue
            rows[term_id] = {
                **row,
                "category": row.get("category") or category.value,
                "confidence": _float(row.get("confidence")),
            }
    return rows


def _float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    if isinstance(value, int | float | str):
        return float(value)
    return default


def _table_label(table: dict[str, Any]) -> str:
    return str(table.get("title") or table.get("filename") or table.get("table_id") or "")


def _build_nodes(
    *,
    run_id: str,
    tables: list[dict[str, Any]],
    terms: list[dict[str, Any]],
    term_outputs: dict[str, dict[str, Any]],
    clusters: list[dict[str, str]],
) -> list[dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for table in tables:
        table_id = str(table["table_id"])
        nodes[table_id] = _node(
            node_id=table_id,
            node_type="table",
            label=_table_label(table),
            properties={
                "filename": table.get("filename"),
                "title": table.get("title"),
                "source_url": table.get("source_url"),
                "parse_status": table.get("parse_status"),
                "row_count": table.get("row_count"),
                "observation_count": table.get("observation_count"),
            },
            run_id=run_id,
        )

    cluster_by_term = {row.get("term_id", ""): row for row in clusters if row.get("term_id")}
    for term in terms:
        term_id = str(term["term_id"])
        output = term_outputs.get(term_id, {})
        cluster = cluster_by_term.get(term_id, {})
        nodes[term_id] = _node(
            node_id=term_id,
            node_type="term",
            label=str(term.get("canonical_term") or output.get("canonical_term") or term_id),
            properties={
                "category": output.get("category"),
                "confidence": output.get("confidence"),
                "table_count": term.get("table_count"),
                "occurrence_count": term.get("occurrence_count"),
                "cluster_id": cluster.get("cluster_id"),
                "domain": cluster.get("domain"),
            },
            run_id=run_id,
        )

    for category in sorted({str(row.get("category")) for row in term_outputs.values()}):
        if not category:
            continue
        nodes[_category_node_id(category)] = _node(
            node_id=_category_node_id(category),
            node_type="category",
            label=category.replace("_", " "),
            properties={"category": category},
            run_id=run_id,
        )

    by_cluster: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clusters:
        if row.get("cluster_id"):
            by_cluster[row["cluster_id"]].append(row)
    for cluster_id, members in by_cluster.items():
        domains = Counter(row.get("domain", "") for row in members if row.get("domain"))
        domain = domains.most_common(1)[0][0] if domains else ""
        representatives = [
            row.get("term", "")
            for row in members
            if str(row.get("is_representative", "")).casefold() == "true"
        ][:10]
        nodes[cluster_id] = _node(
            node_id=cluster_id,
            node_type="cluster",
            label=domain or cluster_id,
            properties={
                "cluster_id": cluster_id,
                "domain": domain,
                "size": len(members),
                "representatives": representatives,
            },
            run_id=run_id,
        )
        if domain:
            domain_id = _domain_node_id(domain)
            nodes[domain_id] = _node(
                node_id=domain_id,
                node_type="domain",
                label=domain,
                properties={"domain": domain},
                run_id=run_id,
            )

    return sorted(nodes.values(), key=lambda row: (str(row["node_type"]), str(row["label"])))


def _base_edges(
    *,
    run_id: str,
    table_terms: list[dict[str, Any]],
    term_outputs: dict[str, dict[str, Any]],
    clusters: list[dict[str, str]],
    relations: list[dict[str, str]],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for row in table_terms:
        table_id = str(row["table_id"])
        term_id = str(row["term_id"])
        occurrence_ids = _json_list(row.get("occurrence_ids_json"))
        edges.append(
            _edge(
                edge_id=stable_id("graph", "table_contains_term", table_id, term_id),
                source_id=table_id,
                target_id=term_id,
                edge_type="table_contains_term",
                weight=1.0,
                directed=True,
                derived=False,
                evidence_ids=occurrence_ids,
                properties={
                    "occurrence_count": row.get("occurrence_count"),
                    "source_areas": _json_list(row.get("source_areas_json")),
                },
                run_id=run_id,
            )
        )

    for term_id, row in term_outputs.items():
        category = str(row.get("category") or "")
        if not category:
            continue
        edges.append(
            _edge(
                edge_id=stable_id("graph", "term_classified_as", term_id, category),
                source_id=term_id,
                target_id=_category_node_id(category),
                edge_type="term_classified_as",
                weight=_float(row.get("confidence"), 1.0),
                directed=True,
                derived=False,
                evidence_ids=_json_list(row.get("occurrence_ids_json")),
                properties={
                    "category": category,
                    "variant": row.get("variant"),
                    "evidence": row.get("evidence"),
                    "protected": row.get("protected"),
                },
                run_id=run_id,
            )
        )

    seen_clusters: set[tuple[str, str]] = set()
    for row in clusters:
        term_id = row.get("term_id", "")
        cluster_id = row.get("cluster_id", "")
        if term_id and cluster_id:
            edges.append(
                _edge(
                    edge_id=stable_id("graph", "measure_member_of_cluster", term_id, cluster_id),
                    source_id=term_id,
                    target_id=cluster_id,
                    edge_type="measure_member_of_cluster",
                    weight=_float(row.get("membership_probability"), 1.0),
                    directed=True,
                    derived=False,
                    properties={
                        "domain": row.get("domain"),
                        "is_representative": row.get("is_representative"),
                        "labeling_method": row.get("labeling_method"),
                    },
                    run_id=run_id,
                )
            )
        domain = row.get("domain", "")
        key = (cluster_id, domain)
        if cluster_id and domain and key not in seen_clusters:
            seen_clusters.add(key)
            edges.append(
                _edge(
                    edge_id=stable_id("graph", "cluster_belongs_to_domain", cluster_id, domain),
                    source_id=cluster_id,
                    target_id=_domain_node_id(domain),
                    edge_type="cluster_belongs_to_domain",
                    weight=1.0,
                    directed=True,
                    derived=False,
                    properties={"domain": domain},
                    run_id=run_id,
                )
            )

    for row in relations:
        source_id = row.get("source_term_id", "")
        target_id = row.get("target_term_id", "")
        relation_type = row.get("relation_type", "")
        if not source_id or not target_id or not relation_type:
            continue
        edges.append(
            _edge(
                edge_id=row.get("relation_id")
                or stable_id("graph", relation_type, source_id, target_id),
                source_id=source_id,
                target_id=target_id,
                edge_type=relation_type,
                weight=_float(row.get("confidence"), 0.0),
                directed=True,
                derived=False,
                evidence_ids=_json_list(row.get("evidence_ids_json")),
                properties={
                    "source_term": row.get("source_term"),
                    "target_term": row.get("target_term"),
                    "confidence": _float(row.get("confidence"), 0.0),
                    "generation_methods": _json_list(row.get("generation_methods_json")),
                    "evidence": _json_dict(row.get("evidence")),
                },
                run_id=run_id,
            )
        )
    return edges


def _table_relation_edges(
    *,
    config: AppConfig,
    run_id: str,
    table_terms: list[dict[str, Any]],
    term_outputs: dict[str, dict[str, Any]],
    clusters: list[dict[str, str]],
    relations: list[dict[str, str]],
) -> list[dict[str, Any]]:
    category_by_term = {
        term_id: str(row.get("category") or "")
        for term_id, row in term_outputs.items()
    }
    domain_by_term = {
        row.get("term_id", ""): row.get("domain", "")
        for row in clusters
        if row.get("term_id") and row.get("domain")
    }
    tables_by_measure: dict[str, set[str]] = defaultdict(set)
    table_domains: dict[str, set[str]] = defaultdict(set)
    table_measures: dict[str, set[str]] = defaultdict(set)
    for row in table_terms:
        term_id = str(row["term_id"])
        if category_by_term.get(term_id) != "measure":
            continue
        table_id = str(row["table_id"])
        tables_by_measure[term_id].add(table_id)
        table_measures[table_id].add(term_id)
        domain = domain_by_term.get(term_id)
        if domain:
            table_domains[table_id].add(domain)

    components: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "shared_measure_count": 0,
            "shared_domains": set(),
            "best_measure_relation_confidence": 0.0,
            "measure_relation_ids": set(),
        }
    )
    for measure_id, table_ids in tables_by_measure.items():
        for left, right in combinations(sorted(table_ids), 2):
            pair = (left, right)
            components[pair]["shared_measure_count"] += 1
            domains = table_domains[left] & table_domains[right]
            components[pair]["shared_domains"].update(domains)
            components[pair].setdefault("shared_measure_ids", set()).add(measure_id)

    for domain in sorted({item for domains in table_domains.values() for item in domains}):
        domain_tables = sorted(
            table_id for table_id, domains in table_domains.items() if domain in domains
        )
        for left, right in combinations(domain_tables, 2):
            components[(left, right)]["shared_domains"].add(domain)

    for relation in relations:
        source = relation.get("source_term_id", "")
        target = relation.get("target_term_id", "")
        if not source or not target:
            continue
        confidence = _float(relation.get("confidence"), 0.0)
        for left in tables_by_measure.get(source, set()):
            for right in tables_by_measure.get(target, set()):
                if left == right:
                    continue
                ordered_pair = sorted((left, right))
                pair = (ordered_pair[0], ordered_pair[1])
                if confidence > components[pair]["best_measure_relation_confidence"]:
                    components[pair]["best_measure_relation_confidence"] = confidence
                components[pair]["measure_relation_ids"].add(relation.get("relation_id", ""))

    scored: list[tuple[float, str, str, dict[str, Any]]] = []
    for (left, right), component in components.items():
        shared_measure_count = int(component["shared_measure_count"])
        shared_domains = sorted(str(item) for item in component["shared_domains"])
        best_relation = float(component["best_measure_relation_confidence"])
        shared_measure_score = min(1.0, shared_measure_count / 3.0)
        shared_domain_score = min(1.0, len(shared_domains) / 2.0)
        weight = (
            (0.45 * shared_measure_score)
            + (0.20 * shared_domain_score)
            + (0.35 * best_relation)
        )
        if weight < config.knowledge_graph.min_table_relation_weight:
            continue
        properties = {
            "components": {
                "shared_measure_count": shared_measure_count,
                "shared_domain_count": len(shared_domains),
                "best_measure_relation_confidence": round(best_relation, 6),
            },
            "shared_measure_ids": sorted(
                str(item) for item in component.get("shared_measure_ids", set())
            )[:25],
            "shared_domains": shared_domains,
            "measure_relation_ids": sorted(
                str(item) for item in component["measure_relation_ids"] if item
            )[:25],
        }
        scored.append((weight, left, right, properties))

    accepted: list[dict[str, Any]] = []
    degree: Counter[str] = Counter()
    max_edges = config.knowledge_graph.max_table_edges_per_table
    for weight, left, right, properties in sorted(
        scored,
        key=lambda item: (-item[0], item[1], item[2]),
    ):
        if degree[left] >= max_edges or degree[right] >= max_edges:
            continue
        degree[left] += 1
        degree[right] += 1
        accepted.append(
            _edge(
                edge_id=stable_id("graph", "table_related_to_table", left, right),
                source_id=left,
                target_id=right,
                edge_type="table_related_to_table",
                weight=weight,
                directed=False,
                derived=True,
                properties=properties,
                run_id=run_id,
            )
        )
    return accepted


def validate_graph_rows(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    """Validate graph node and edge integrity before publishing artifacts."""

    node_ids = [str(row.get("node_id") or "") for row in nodes]
    if len(node_ids) != len(set(node_ids)):
        duplicates = [node_id for node_id, count in Counter(node_ids).items() if count > 1]
        raise ValueError("Duplicate graph node IDs: " + ", ".join(duplicates[:10]))
    node_set = set(node_ids)
    edge_ids = [str(row.get("edge_id") or "") for row in edges]
    if len(edge_ids) != len(set(edge_ids)):
        duplicates = [edge_id for edge_id, count in Counter(edge_ids).items() if count > 1]
        raise ValueError("Duplicate graph edge IDs: " + ", ".join(duplicates[:10]))
    unknown_edges = [
        str(row.get("edge_id") or "")
        for row in edges
        if row.get("source_id") not in node_set or row.get("target_id") not in node_set
    ]
    if unknown_edges:
        raise ValueError("Graph edges reference unknown nodes: " + ", ".join(unknown_edges[:10]))
    self_table_edges = [
        str(row.get("edge_id") or "")
        for row in edges
        if row.get("edge_type") == "table_related_to_table"
        and row.get("source_id") == row.get("target_id")
    ]
    if self_table_edges:
        raise ValueError(
            "Derived table graph edges cannot be self edges: "
            + ", ".join(self_table_edges[:10])
        )


def _summary(
    *,
    run_id: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_path: Path,
    edge_path: Path,
) -> dict[str, Any]:
    node_counts = Counter(str(row["node_type"]) for row in nodes)
    edge_counts = Counter(str(row["edge_type"]) for row in edges)
    return {
        "run_id": run_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_type_counts": dict(sorted(node_counts.items())),
        "edge_type_counts": dict(sorted(edge_counts.items())),
        "files": {
            "nodes": {
                "path": str(node_path),
                "size_bytes": node_path.stat().st_size,
                "md5": file_md5(node_path),
            },
            "edges": {
                "path": str(edge_path),
                "size_bytes": edge_path.stat().st_size,
                "md5": file_md5(edge_path),
            },
        },
    }


def run_knowledge_graph(
    config: AppConfig,
    *,
    resources: tuple[ResourceRecord, ...] = (),
) -> tuple[dict[str, Path], Path, dict[str, Any]]:
    """Build materialized knowledge graph artifacts from pipeline outputs."""

    manifest = create_manifest(config, "knowledge_graph")
    run_id = manifest.run_id
    tables = read_parquet_rows(config.paths.processed_dir / "tables.parquet")
    terms = read_parquet_rows(config.paths.processed_dir / "vocabulary.parquet")
    table_terms = read_parquet_rows(config.paths.processed_dir / "table_vocabulary.parquet")
    term_outputs = _load_term_outputs(config)
    clusters = _read_csv(config.paths.outputs_dir / "measure_clusters.csv", required=True)
    relations = _read_csv(config.paths.outputs_dir / "measure_relations.csv", required=True)

    nodes = _build_nodes(
        run_id=run_id,
        tables=tables,
        terms=terms,
        term_outputs=term_outputs,
        clusters=clusters,
    )
    edges = [
        *_base_edges(
            run_id=run_id,
            table_terms=table_terms,
            term_outputs=term_outputs,
            clusters=clusters,
            relations=relations,
        ),
        *_table_relation_edges(
            config=config,
            run_id=run_id,
            table_terms=table_terms,
            term_outputs=term_outputs,
            clusters=clusters,
            relations=relations,
        ),
    ]
    validate_graph_rows(nodes, edges)

    node_path = write_parquet_rows(
        nodes,
        config.paths.processed_dir / "knowledge_graph_nodes.parquet",
    )
    edge_path = write_parquet_rows(
        edges,
        config.paths.processed_dir / "knowledge_graph_edges.parquet",
    )
    output_dir = config.paths.outputs_dir / "knowledge_graph" / run_id
    summary_payload = _summary(
        run_id=run_id,
        nodes=nodes,
        edges=edges,
        node_path=node_path,
        edge_path=edge_path,
    )
    summary_path = _write_json(summary_payload, output_dir / "knowledge_graph_summary.json")
    current_path = _write_json(
        {
            "run_id": run_id,
            "nodes": str(node_path),
            "edges": str(edge_path),
            "summary": str(summary_path),
        },
        config.paths.outputs_dir / "knowledge_graph" / "current_graph.json",
    )
    artifacts = {
        "nodes": node_path,
        "edges": edge_path,
        "summary": summary_path,
        "current_graph": current_path,
    }
    completed = complete_manifest(
        manifest,
        resources=tuple(record.resource_id for record in resources),
        artifacts=tuple(str(path) for path in artifacts.values()),
    )
    manifest_path = write_manifest(
        completed,
        config.paths.outputs_dir / "manifests" / f"{manifest.run_id}_knowledge_graph.json",
    )
    diagnostics = {
        "run_id": run_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_type_counts": summary_payload["node_type_counts"],
        "edge_type_counts": summary_payload["edge_type_counts"],
    }
    return artifacts, manifest_path, diagnostics
