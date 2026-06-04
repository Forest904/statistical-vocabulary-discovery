"""PEARL-small FAISS semantic indexing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from statvocab.classification_features import read_parquet_rows, write_parquet_rows
from statvocab.config import AppConfig


@dataclass(frozen=True)
class SemanticHit:
    """One semantic retrieval candidate."""

    table_id: str
    score: float


def _import_dependencies() -> tuple[Any, Any, Any]:
    try:
        numpy = import_module("numpy")
        faiss = import_module("faiss")
        sentence_transformers = import_module("sentence_transformers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Semantic retrieval requires the ML extra, including faiss-cpu and "
            "sentence-transformers. Install with `python -m pip install -e .[ml]`."
        ) from exc
    return numpy, faiss, sentence_transformers.SentenceTransformer


class PearlEmbedder:
    """Lazy PEARL-small embedding wrapper."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            _numpy, _faiss, sentence_transformer = _import_dependencies()
            model_cache = self._config.paths.cache_dir / "models"
            model_cache.mkdir(parents=True, exist_ok=True)
            self._model = sentence_transformer(
                self._config.classification.pearl_model,
                revision=self._config.classification.pearl_revision,
                cache_folder=str(model_cache),
            )
        return self._model

    def encode(self, texts: list[str]) -> Any:
        numpy, _faiss, _sentence_transformer = _import_dependencies()
        if not texts:
            return numpy.empty((0, 0), dtype="float32")
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self._config.search.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return numpy.asarray(embeddings, dtype="float32")


def build_semantic_index(
    config: AppConfig,
    rows: list[dict[str, object]],
    *,
    index_path: Path,
    metadata_path: Path,
) -> tuple[Path, Path, int]:
    """Build a FAISS CPU index over normalized table-document embeddings."""

    numpy, faiss, _sentence_transformer = _import_dependencies()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    texts = [str(row.get("semantic_text") or "") for row in rows]
    matrix = PearlEmbedder(config).encode(texts)
    dimension = int(matrix.shape[1]) if len(matrix.shape) == 2 and matrix.shape[0] else 0
    if dimension == 0:
        index = faiss.IndexFlatIP(1)
        faiss.write_index(index, str(index_path))
        metadata = []
    else:
        matrix = numpy.ascontiguousarray(matrix, dtype="float32")
        index = faiss.IndexFlatIP(dimension)
        index.add(matrix)
        faiss.write_index(index, str(index_path))
        metadata = [
            {
                "vector_position": index,
                "table_id": row["table_id"],
                "semantic_text": row.get("semantic_text") or "",
            }
            for index, row in enumerate(rows)
        ]
    return index_path, write_parquet_rows(metadata, metadata_path), dimension


def search_semantic(
    config: AppConfig,
    *,
    index_path: Path,
    metadata_path: Path,
    query: str,
    top_k: int,
    embedder: PearlEmbedder | None = None,
) -> list[SemanticHit]:
    """Search the FAISS index with a PEARL-small query embedding."""

    if not query.strip() or not index_path.exists() or not metadata_path.exists():
        return []
    numpy, faiss, _sentence_transformer = _import_dependencies()
    metadata = read_parquet_rows(metadata_path)
    if not metadata:
        return []
    by_position = {
        int(row["vector_position"]): str(row["table_id"])
        for row in metadata
    }
    index = faiss.read_index(str(index_path))
    query_matrix = (embedder or PearlEmbedder(config)).encode([query])
    if query_matrix.shape[1] != index.d:
        return []
    query_matrix = numpy.ascontiguousarray(query_matrix, dtype="float32")
    scores, positions = index.search(query_matrix, min(top_k, len(metadata)))
    hits: list[SemanticHit] = []
    for score, position in zip(
        cast(list[float], list(scores[0])),
        cast(list[int], list(positions[0])),
        strict=True,
    ):
        if int(position) < 0:
            continue
        table_id = by_position.get(int(position))
        if table_id is None:
            continue
        hits.append(SemanticHit(table_id=table_id, score=float(score)))
    return hits
