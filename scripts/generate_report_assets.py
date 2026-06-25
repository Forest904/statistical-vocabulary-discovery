"""Generate static figures for the final STAR assignment report."""

from __future__ import annotations

import csv
import json
import math
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
ASSET_DIR = REPORT_DIR / "assets"

BLUE = "#2F5D7C"
TEAL = "#2F7D73"
GOLD = "#B68A35"
ROSE = "#A85C5C"
GREEN = "#6A8D3B"
GRAY = "#6A6F73"
DARK = "#1F2933"
LIGHT = "#F5F7FA"
LINE = "#D8DEE6"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _setup() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "axes.edgecolor": LINE,
            "axes.linewidth": 0.8,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "text.color": DARK,
        }
    )


def _save(fig: plt.Figure, filename: str) -> Path:
    path = ASSET_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    return path


def _bar_labels(ax: plt.Axes, *, fmt: str = "{:,.0f}", padding: float = 0.01) -> None:
    xmax = ax.get_xlim()[1]
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(
            width + xmax * padding,
            patch.get_y() + patch.get_height() / 2,
            fmt.format(width),
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color=DARK,
        )


def _bold_axes(ax: plt.Axes) -> None:
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def _draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    *,
    color: str,
    title_size: int = 9,
    body_size: int = 7,
) -> None:
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.45,
        edgecolor=color,
        facecolor="#FFFFFF",
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h - min(0.030, h * 0.18),
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=color,
    )
    ax.text(
        x + w / 2,
        y + h * 0.39,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        fontweight="bold",
        color=DARK,
        linespacing=1.35,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    rad: float = 0.0,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.35,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


def assignment_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        ((0.04, 0.70), "1. Time D(t)", "Top-line intervals\nfrom table headers", BLUE),
        ((0.28, 0.70), "2. Strings S(t)", "Leftmost nonnumeric\nmetadata cells", TEAL),
        ((0.52, 0.70), "3. Geo(t)", "NUTS/GEO matches\nremoved from S(t)", GREEN),
        ((0.76, 0.70), "4. Titles", "Remove time and geo\nfrom table titles", GOLD),
        ((0.28, 0.42), "5. Vocabulary V", "Union of cleaned\ncell and title terms", BLUE),
        ((0.52, 0.42), "6. Partition", "M, N, A, U\nplus Other", TEAL),
        ((0.28, 0.15), "7. Domains", "Cluster measures\ninto domains", GREEN),
        ((0.52, 0.15), "8. Relations", "Typed measure\ncandidate graph", ROSE),
        ((0.76, 0.15), "Submission", "CSV artifacts,\nmetrics, report", DARK),
    ]
    for xy, title, body, color in boxes:
        _draw_box(ax, xy, (0.18, 0.15), title, body, color=color)

    _arrow(ax, (0.22, 0.775), (0.28, 0.775))
    _arrow(ax, (0.46, 0.775), (0.52, 0.775))
    _arrow(ax, (0.70, 0.775), (0.76, 0.775))
    _arrow(ax, (0.76, 0.70), (0.46, 0.57))
    _arrow(ax, (0.37, 0.70), (0.37, 0.57))
    _arrow(ax, (0.46, 0.495), (0.52, 0.495))
    _arrow(ax, (0.52, 0.42), (0.46, 0.30))
    _arrow(ax, (0.61, 0.42), (0.61, 0.30))
    _arrow(ax, (0.46, 0.225), (0.52, 0.225))
    _arrow(ax, (0.70, 0.225), (0.76, 0.225))

    ax.text(
        0.5,
        0.965,
        "Assignment Pipeline for the 2,000-Table Core Run",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.900,
        "Every figure and metric in the report is derived from the validated core corpus artifacts.",
        ha="center",
        va="center",
        fontsize=9,
        color=GRAY,
    )
    return _save(fig, "assignment_pipeline.png")


def repo_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    groups = [
        ((0.05, 0.68), "Inputs", "STAR tables\nTitles\nNUTS/GEO\nGold labels", BLUE),
        ((0.31, 0.68), "Core Package", "resources/ingest\nextract/vocabulary\nclassify/cluster\nrelations/search", TEAL),
        ((0.58, 0.68), "Artifacts", "Parquet tables\nRequired CSVs\nMetrics JSON\nSearch indexes", GREEN),
        ((0.58, 0.37), "Product Layer", "FastAPI routes\nReact pages\nGraph explorer\nReview UI", GOLD),
        ((0.31, 0.37), "Evaluation", "Extraction audit\nClassification gold\nCluster review\nRelation review", ROSE),
        ((0.05, 0.37), "Reproducibility", "configs/*.yaml\nrun manifests\ncore release manifest\ntests", DARK),
        ((0.31, 0.08), "Report", "Final PDF\nFigures\nLimitations\nFuture scale work", BLUE),
        ((0.58, 0.08), "Engineering Surface", "CLI commands\nDocker compose\nnotebooks\nCI-ready tests", TEAL),
    ]
    for xy, title, body, color in groups:
        _draw_box(ax, xy, (0.21, 0.18), title, body, color=color, body_size=7)

    _arrow(ax, (0.26, 0.77), (0.31, 0.77))
    _arrow(ax, (0.52, 0.77), (0.58, 0.77))
    _arrow(ax, (0.69, 0.68), (0.69, 0.55))
    _arrow(ax, (0.58, 0.46), (0.52, 0.46))
    _arrow(ax, (0.31, 0.46), (0.26, 0.46))
    _arrow(ax, (0.42, 0.37), (0.42, 0.26))
    _arrow(ax, (0.69, 0.37), (0.69, 0.26))
    _arrow(ax, (0.58, 0.17), (0.52, 0.17))
    _arrow(ax, (0.26, 0.55), (0.31, 0.68))
    _arrow(ax, (0.42, 0.68), (0.42, 0.55))

    ax.text(
        0.5,
        0.982,
        "High-Level Repository Architecture",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.928,
        "The project separates reproducible research artifacts from the supplementary product surface.",
        ha="center",
        va="center",
        fontsize=9,
        color=GRAY,
    )
    return _save(fig, "repo_architecture.png")


def category_partition() -> Path:
    metrics = _read_json(REPORT_DIR / "classification_metrics.json")
    labels = [
        ("Measures", metrics["category_counts"]["measure"], BLUE),
        ("Dimension\nnames", metrics["category_counts"]["dimension_name"], TEAL),
        ("Dimension\nvalues", metrics["category_counts"]["dimension_value"], GREEN),
        ("Units", metrics["category_counts"]["unit"], GOLD),
        ("Other /\nambiguous", metrics["category_counts"]["other_ambiguous"], GRAY),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    names = [item[0] for item in labels]
    counts = [item[1] for item in labels]
    colors = [item[2] for item in labels]
    bars = ax.bar(names, counts, color=colors, width=0.68)
    ax.set_title("Vocabulary Partition for the Core Submission")
    ax.set_ylabel("Distinct terms")
    ax.grid(axis="y", color=LINE, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.text(
        0.0,
        -0.20,
        "The conservative Other bucket prevents low-confidence terms from polluting required files.",
        transform=ax.transAxes,
        ha="left",
        color=GRAY,
        fontsize=9,
        fontweight="bold",
    )
    _bold_axes(ax)
    return _save(fig, "category_partition.png")


def classification_metrics() -> Path:
    metrics = _read_json(REPORT_DIR / "classification_metrics.json")
    sources = [
        ("Random audit", metrics["source_metrics"]["random_sample"]),
        ("Targeted reclaim", metrics["targeted_reclaim_metrics"]),
    ]
    metric_names = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("weighted_f1", "Weighted-F1")]
    split_names = [("metrics", "All"), ("validation_metrics", "Validation"), ("final_test_metrics", "Final test")]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    colors = [BLUE, TEAL, GOLD]
    x = range(len(metric_names))
    width = 0.24
    for ax, (title, payload) in zip(axes, sources, strict=True):
        for offset, (split_key, split_label) in enumerate(split_names):
            values = [payload[split_key][key] for key, _ in metric_names]
            positions = [idx + (offset - 1) * width for idx in x]
            ax.bar(positions, values, width=width, label=split_label, color=colors[offset])
        ax.set_title(title)
        ax.set_xticks(list(x), [label for _, label in metric_names])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color=LINE, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        _bold_axes(ax)
    axes[0].set_ylabel("Score")
    axes[1].legend(loc="upper right", frameon=False)
    for text in axes[1].get_legend().get_texts():
        text.set_fontweight("bold")
    fig.suptitle("Classification Quality: Headline Audit vs. Stress Test", fontweight="bold", y=1.02)
    fig.text(
        0.05,
        -0.02,
        "The random audit is the headline submission metric; targeted reclaim intentionally probes hard Other cases.",
        color=GRAY,
        fontsize=9,
        fontweight="bold",
    )
    return _save(fig, "classification_metrics.png")


def cluster_domain_distribution() -> Path:
    metrics = _read_json(REPORT_DIR / "clustering_metrics.json")
    counts = dict(metrics["domain_distribution"])
    ordered = sorted(counts.items(), key=lambda item: item[1])
    names = [textwrap.fill(name, 28) for name, _ in ordered]
    values = [value for _, value in ordered]
    colors = [ROSE if "cross-domain" in name else BLUE for name, _ in ordered]
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.barh(names, values, color=colors)
    ax.set_title("Measure Domain Distribution")
    ax.set_xlabel("Measures")
    ax.grid(axis="x", color=LINE, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    _bar_labels(ax)
    _bold_axes(ax)
    return _save(fig, "cluster_domain_distribution.png")


def cluster_review_quality() -> Path:
    review = _read_json(REPORT_DIR / "clustering_metrics.json")["manual_review"]
    review_rows = _read_csv(ROOT / review["review_sample"])
    coherence = {
        score: sum(1 for row in review_rows if row.get("coherence_score", "").strip() == score)
        for score in ["0", "1", "2"]
    }
    domain = review["domain_label_quality_distribution"]
    representative = review["representative_quality_distribution"]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.5))
    panels = [
        ("Coherence score", coherence, [ROSE, GOLD, GREEN]),
        ("Domain label quality", domain, [GREEN, GOLD, ROSE]),
        ("Representative quality", representative, [GREEN, ROSE]),
    ]
    for ax, (title, data, colors) in zip(axes, panels, strict=True):
        names = list(data)
        values = [data[name] for name in names]
        bars = ax.bar(names, values, color=colors[: len(names)], width=0.62)
        ax.set_title(title)
        ax.set_ylabel("Reviewed rows")
        ax.grid(axis="y", color=LINE, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.03,
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        _bold_axes(ax)
    fig.suptitle("Completed Step 7 Manual Review", fontweight="bold", y=1.02)
    fig.text(
        0.05,
        -0.03,
        (
            f"{review['completed_count']} reviewed rows; mean coherence "
            f"{review['mean_coherence']:.3f}/2; domain-label accuracy "
            f"{review['domain_label_accuracy']:.3f}."
        ),
        color=GRAY,
        fontsize=9,
        fontweight="bold",
    )
    return _save(fig, "cluster_review_quality.png")


def relation_type_distribution() -> Path:
    counts = _read_json(REPORT_DIR / "relations_metrics.json")["relation_type_distribution"]
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    names = [name.replace("_", "\n") for name, _ in ordered]
    values = [value for _, value in ordered]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(names, values, color=[BLUE, TEAL, GOLD], width=0.62)
    ax.set_title("Submitted Measure Relationship Types")
    ax.set_ylabel("Candidate relationships")
    ax.grid(axis="y", color=LINE, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    _bold_axes(ax)
    return _save(fig, "relation_type_distribution.png")


def pipeline_runtime() -> Path:
    stages = _read_json(REPORT_DIR / "performance_metrics.json")["stages"]
    display_order = [
        ("ingest", "Ingest"),
        ("extraction", "Extract"),
        ("classification", "Classify"),
        ("clustering", "Cluster"),
        ("relations", "Relations"),
        ("search_index", "Search index"),
    ]
    data = [
        (label, float(stages[key]["wall_clock_seconds"]))
        for key, label in display_order
        if stages.get(key, {}).get("wall_clock_seconds") is not None
    ]
    data.sort(key=lambda item: item[1])
    names = [label for label, _ in data]
    values = [seconds for _, seconds in data]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.barh(names, values, color=BLUE)
    ax.set_title("Core Pipeline Runtime by Stage")
    ax.set_xlabel("Wall-clock seconds")
    ax.grid(axis="x", color=LINE, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    _bar_labels(ax, fmt="{:,.1f}", padding=0.015)
    _bold_axes(ax)
    total = sum(values)
    ax.text(
        0.0,
        -0.20,
        f"Measured on the local core run; chart excludes interactive review time. Total shown: {total:.1f}s.",
        transform=ax.transAxes,
        ha="left",
        color=GRAY,
        fontsize=9,
        fontweight="bold",
    )
    return _save(fig, "pipeline_runtime.png")


def main() -> None:
    _setup()
    generated = [
        assignment_pipeline(),
        repo_architecture(),
        category_partition(),
        classification_metrics(),
        cluster_domain_distribution(),
        cluster_review_quality(),
        relation_type_distribution(),
        pipeline_runtime(),
    ]
    print("Generated report assets:")
    for path in generated:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
