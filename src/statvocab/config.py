"""Configuration loading for StatVocab pipeline stages."""

from __future__ import annotations

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

    @property
    def config_id_parts(self) -> tuple[str, str, int]:
        return (self.config_name, self.corpus.name, self.random_seed)


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML config file into the stable Pydantic contract."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    return AppConfig.model_validate(raw)
