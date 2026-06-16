"""Read-only summaries for core artifact hardening work."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _json_list(value: str | None) -> list[Any]:
    if not value:
        return []
    payload = json.loads(value)
    return payload if isinstance(payload, list) else []


def ambiguity_summary() -> dict[str, Any]:
    rows = _read_csv(Path("outputs/other_ambiguous.csv"))
    occurrences = pq.read_table(
        "data/processed/term_occurrences.parquet",
        columns=["term_id", "table_id", "source"],
    ).to_pylist()
    by_term: dict[str, dict[str, set[str]]] = {}
    for occurrence in occurrences:
        term_id = str(occurrence["term_id"])
        item = by_term.setdefault(term_id, {"tables": set(), "sources": set()})
        item["tables"].add(str(occurrence["table_id"]))
        item["sources"].add(str(occurrence["source"]))
    ranked = []
    source_counts: Counter[str] = Counter()
    for row in rows:
        term_id = row["term_id"]
        item = by_term.get(term_id, {"tables": set(), "sources": set()})
        for source in item["sources"]:
            source_counts[source] += 1
        ranked.append(
            {
                "term_id": term_id,
                "canonical_term": row["canonical_term"],
                "table_count": len(item["tables"]),
                "sources": sorted(item["sources"]),
            }
        )
    ranked.sort(key=lambda item: (-int(item["table_count"]), item["canonical_term"].casefold()))
    return {
        "other_ambiguous_count": len(rows),
        "source_distribution": dict(sorted(source_counts.items())),
        "top_by_table_coverage": ranked[:50],
    }


def relation_summary() -> dict[str, Any]:
    metrics = json.loads(Path("report/relations_metrics.json").read_text(encoding="utf-8"))
    manual = metrics.get("manual_review", {})
    return {
        "relation_count": metrics.get("relation_count"),
        "precision_at_100": manual.get("precision_at_100"),
        "typed_accuracy": manual.get("typed_accuracy"),
        "false_positive_taxonomy": manual.get("false_positive_taxonomy", {}),
    }


def unclustered_summary() -> dict[str, Any]:
    clusters = _read_csv(Path("outputs/measure_clusters.csv"))
    unclustered = [
        row
        for row in clusters
        if row.get("cluster_id", "").casefold() in {"unclustered", "noise"}
        or row.get("domain", "").casefold() in {"unclustered", "noise"}
    ]
    domains = Counter(row.get("domain", "") or "unknown" for row in clusters)
    return {
        "measure_cluster_rows": len(clusters),
        "unclustered_count": len(unclustered),
        "domain_distribution": dict(domains.most_common()),
        "unclustered_examples": unclustered[:50],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("area", choices=["ambiguity", "relations", "unclustered"])
    args = parser.parse_args()
    payload = {
        "ambiguity": ambiguity_summary,
        "relations": relation_summary,
        "unclustered": unclustered_summary,
    }[args.area]()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
