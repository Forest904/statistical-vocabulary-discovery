"""Optional local embedding support for Milestone 3 classification."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from statvocab.classification_features import write_parquet_rows
from statvocab.config import AppConfig


def _sentence_transformer_class() -> Any:
    try:
        module = import_module("sentence_transformers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Local-hybrid classification with completed gold labels requires the ML extra "
            "and sentence-transformers. Install with `python -m pip install -e .[ml]`."
        ) from exc
    return module.SentenceTransformer


def generate_term_embeddings(config: AppConfig, feature_rows: list[dict[str, Any]]) -> Path:
    """Generate and cache PEARL-small embeddings for all feature rows."""

    sentence_transformer = _sentence_transformer_class()
    model_cache = config.paths.cache_dir / "models"
    model_cache.mkdir(parents=True, exist_ok=True)
    model = sentence_transformer(
        config.classification.pearl_model,
        revision=config.classification.pearl_revision,
        cache_folder=str(model_cache),
    )
    terms = [str(row["canonical_term"]) for row in feature_rows]
    embeddings = model.encode(
        terms,
        batch_size=config.classification.embedding_batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    rows = [
        {
            "term_id": row["term_id"],
            "canonical_term": row["canonical_term"],
            "model": config.classification.pearl_model,
            "revision": config.classification.pearl_revision,
            "embedding": [float(value) for value in embedding],
        }
        for row, embedding in zip(feature_rows, embeddings, strict=True)
    ]
    return write_parquet_rows(rows, config.paths.processed_dir / "term_embeddings.parquet")
