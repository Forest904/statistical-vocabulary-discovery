"""Configuration loading for StatVocab pipeline stages."""

from __future__ import annotations

import json
from hashlib import blake2b
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class CorpusConfig(BaseModel):
    """Input corpus selection and expected resource metadata."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["fixture", "core", "full"]
    archive_name: str | None = None
    expected_table_count: int = Field(gt=0)
    expected_md5: str | None = None
    fixture_path: Path | None = None


class PathsConfig(BaseModel):
    """Project paths used by pipeline stages."""

    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    external_dir: Path = Path("data/external")
    processed_dir: Path = Path("data/processed")
    outputs_dir: Path = Path("outputs")
    reports_dir: Path = Path("report")
    prompts_dir: Path = Path("prompts")
    cache_dir: Path = Path(".cache/statvocab")


class EvaluationConfig(BaseModel):
    """Evaluation defaults shared across milestones."""

    model_config = ConfigDict(extra="forbid")

    fixture_corpus: Path = Path("data/fixtures/eurostat_star_edge_cases")
    extraction_gold_dir: Path = Path("data/gold")
    random_seed: int = 42
    extraction_review_tables: int = Field(default=50, gt=0)
    vocabulary_gold_terms: int = Field(default=500, gt=0)


class ExtractionConfig(BaseModel):
    """Milestone 2 extraction defaults."""

    model_config = ConfigDict(extra="forbid")

    geography_variant: Literal["nuts", "enhanced"] = "enhanced"
    review_sample_size: int = Field(default=50, gt=0)
    min_year: int = 1900
    max_year: int = 2100
    nuts_2024_path: Path = Path("data/external/NUTS_AT_2024.csv")
    eurostat_geo_codelist_path: Path = Path("data/external/eurostat_geo_codelist.xml")
    eurostat_geo_codelist_version: str = "GEO 14.0"
    parallel_workers: int = Field(default=1, gt=0)
    parallel_chunk_size: int = Field(default=25, gt=0)


class ClassificationConfig(BaseModel):
    """Milestone 3 semantic classification defaults."""

    model_config = ConfigDict(extra="forbid")

    gold_sample_size: int = Field(default=500, gt=0)
    blind_relabel_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    train_dev_fraction: float = Field(default=0.60, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.20, gt=0.0, lt=1.0)
    final_test_fraction: float = Field(default=0.20, gt=0.0, lt=1.0)
    pearl_model: str = "Lihuchen/pearl_small"
    pearl_revision: str = "0d29fb4a61ec2a11b60e8b078664389eb915286b"
    embedding_batch_size: int = Field(default=64, gt=0)
    abstention_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    semantic_neighbor_k: int = Field(default=5, gt=0)
    semantic_hybrid_enabled: bool = True
    threshold_objective: Literal["macro_f1", "macro_f1_with_recall_bonus"] = (
        "macro_f1_with_recall_bonus"
    )
    classifier_max_iter: int = Field(default=5000, gt=0)
    balance_class_weight: bool = True
    targeted_reclaim_sample_size: int = Field(default=250, gt=0)
    targeted_relabel_size: int = Field(default=25, ge=0)
    targeted_train_dev_count: int = Field(default=150, ge=0)
    targeted_validation_count: int = Field(default=50, ge=0)
    targeted_final_test_count: int = Field(default=50, ge=0)
    non_other_precision_floor: float = Field(default=0.85, ge=0.0, le=1.0)
    min_non_other_precision: float = Field(default=0.85, ge=0.0, le=1.0)
    acceptance_final_macro_drop: float = Field(default=0.02, ge=0.0, le=1.0)
    min_other_reduction_fraction: float = Field(default=0.20, ge=0.0, le=1.0)
    min_measure_recall_for_promotion: float = Field(default=0.35, ge=0.0, le=1.0)


class ClusteringConfig(BaseModel):
    """Milestone 4 measure-domain clustering defaults."""

    model_config = ConfigDict(extra="forbid")

    hdbscan_min_cluster_size: int = Field(default=5, gt=1)
    hdbscan_min_samples: int = Field(default=2, gt=0)
    hdbscan_metric: str = "euclidean"
    agglomerative_distance_threshold: float = Field(default=0.35, gt=0.0)
    agglomerative_linkage: str = "average"
    domain_similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    domain_similarity_margin: float = Field(default=0.03, ge=0.0, le=1.0)
    contextual_features_enabled: bool = True
    contextual_embedding_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    contextual_table_context_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    contextual_lexical_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    min_domain_assignment_score: float = Field(default=0.35, ge=0.0, le=1.0)
    representative_count: int = Field(default=5, gt=0)
    manual_review_sample_size: int = Field(default=100, gt=0)


class RelationsConfig(BaseModel):
    """Milestone 5 measure-to-measure relationship defaults."""

    model_config = ConfigDict(extra="forbid")

    embedding_similarity_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    related_similarity_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    containment_min_extra_tokens: int = Field(default=1, gt=0)
    max_candidates_per_measure: int = Field(default=25, gt=0)
    semantic_neighbor_k: int = Field(default=50, gt=0)
    manual_review_sample_size: int = Field(default=100, gt=0)
    llm_adjudication_enabled: bool = False
    llm_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    submitted_related_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)


class SearchConfig(BaseModel):
    """Milestone 6 search index and retrieval defaults."""

    model_config = ConfigDict(extra="forbid")

    lexical_top_k: int = Field(default=100, gt=0)
    semantic_top_k: int = Field(default=100, gt=0)
    default_limit: int = Field(default=20, gt=0)
    semantic_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    lexical_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    measure_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    dimension_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    geography_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    time_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    min_fused_score: float = Field(default=0.10, ge=0.0, le=1.0)
    development_fraction: float = Field(default=0.60, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.20, gt=0.0, lt=1.0)
    final_test_fraction: float = Field(default=0.20, gt=0.0, lt=1.0)
    embedding_batch_size: int = Field(default=64, gt=0)


class KnowledgeGraphConfig(BaseModel):
    """Knowledge graph construction and bounded-serving defaults."""

    model_config = ConfigDict(extra="forbid")

    min_table_relation_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    max_table_edges_per_table: int = Field(default=20, gt=0)
    default_depth: int = Field(default=1, gt=0)
    max_depth: int = Field(default=2, gt=0)
    max_subgraph_nodes: int = Field(default=250, gt=0)
    max_subgraph_edges: int = Field(default=600, gt=0)


class AppConfig(BaseModel):
    """Top-level typed project configuration."""

    model_config = ConfigDict(extra="forbid")

    config_name: str
    random_seed: int = 42
    corpus: CorpusConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    relations: RelationsConfig = Field(default_factory=RelationsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)

    @property
    def config_id_parts(self) -> tuple[str, str, int]:
        return (self.config_name, self.corpus.name, self.random_seed)

    @property
    def config_summary(self) -> dict[str, str | int]:
        """Return the short human-readable run identity."""

        return {
            "config_name": self.config_name,
            "corpus": self.corpus.name,
            "random_seed": self.random_seed,
        }

    @property
    def config_fingerprint(self) -> str:
        """Return a stable fingerprint over the complete normalized config."""

        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
        return f"cfg_{digest}"


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML config file into the stable Pydantic contract."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    return AppConfig.model_validate(raw)
