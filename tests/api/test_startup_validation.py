from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.app.state import (
    ApiStartupError,
    validate_loaded_artifacts,
    validate_required_artifact_paths,
    validate_search_artifacts,
)
from statvocab.config import AppConfig, load_config
from statvocab.resources import file_md5


def _config(tmp_path: Path) -> AppConfig:
    base = load_config("configs/core.yaml")
    paths = base.paths.model_copy(
        update={
            "outputs_dir": tmp_path / "outputs",
            "processed_dir": tmp_path / "processed",
        }
    )
    return base.model_copy(update={"paths": paths})


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _valid_index(config: AppConfig) -> dict[str, Path]:
    root = config.paths.outputs_dir / "search" / "run_test"
    documents = _write_file(root / "documents.parquet", "documents")
    lexical = _write_file(root / "lexical.sqlite", "lexical")
    semantic = _write_file(root / "semantic.faiss", "semantic")
    metadata = _write_file(root / "semantic_metadata.parquet", "metadata")
    summary = _write_json(
        root / "search_index_summary.json",
        {
            "run_id": "run_test",
            "files": {
                "search_documents": {
                    "path": str(documents),
                    "size_bytes": documents.stat().st_size,
                    "md5": file_md5(documents),
                },
                "lexical_index": {
                    "path": str(lexical),
                    "size_bytes": lexical.stat().st_size,
                    "md5": file_md5(lexical),
                },
                "semantic_index": {
                    "path": str(semantic),
                    "size_bytes": semantic.stat().st_size,
                    "md5": file_md5(semantic),
                },
                "semantic_metadata": {
                    "path": str(metadata),
                    "size_bytes": metadata.stat().st_size,
                    "md5": file_md5(metadata),
                },
            },
        },
    )
    manifest = _write_json(
        root / "index_manifest.json",
        {
            "run_id": "run_test",
            "documents": str(documents),
            "lexical_index": str(lexical),
            "semantic_index": str(semantic),
            "semantic_metadata": str(metadata),
            "summary": str(summary),
        },
    )
    _write_json(
        config.paths.outputs_dir / "search" / "current_index.json",
        {"run_id": "run_test", "index_manifest": str(manifest)},
    )
    return {
        "documents": documents,
        "lexical": lexical,
        "semantic": semantic,
        "metadata": metadata,
        "summary": summary,
        "manifest": manifest,
    }


def test_missing_current_index_fails_with_actionable_error(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(ApiStartupError) as exc_info:
        validate_search_artifacts(config)

    assert exc_info.value.code == "missing_current_index"
    assert "build-search-index" in str(exc_info.value)


def test_missing_index_file_fails_with_actionable_error(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = _valid_index(config)
    paths["semantic"].unlink()

    with pytest.raises(ApiStartupError) as exc_info:
        validate_search_artifacts(config)

    assert exc_info.value.code == "missing_index_file"
    assert exc_info.value.details["key"] == "semantic_index"


def test_stale_index_file_fails_with_checksum_error(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = _valid_index(config)
    paths["lexical"].write_text("changed", encoding="utf-8")

    with pytest.raises(ApiStartupError) as exc_info:
        validate_search_artifacts(config)

    assert exc_info.value.code == "stale_index_file"
    assert exc_info.value.details["expected_md5"] != exc_info.value.details["actual_md5"]


def test_valid_index_artifacts_return_manifest_context(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _valid_index(config)

    payload = validate_search_artifacts(config)

    assert payload["manifest"]["run_id"] == "run_test"
    assert payload["summary"]["run_id"] == "run_test"


def test_missing_required_output_artifact_fails_startup(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("tables.parquet", "vocabulary.parquet", "table_vocabulary.parquet"):
        _write_file(config.paths.processed_dir / filename, "placeholder")

    with pytest.raises(ApiStartupError) as exc_info:
        validate_required_artifact_paths(config)

    assert exc_info.value.code == "missing_artifact"
    assert "measures.csv" in str(exc_info.value)


def test_loaded_artifacts_reject_table_document_mismatch() -> None:
    with pytest.raises(ApiStartupError) as exc_info:
        validate_loaded_artifacts(
            tables={"table_1": {}},
            search_documents={},
            terms={},
            term_outputs={},
            table_terms={},
            clusters=[{"cluster_id": "cluster_1"}],
            cluster_by_term={},
            relations=[],
        )

    assert exc_info.value.code == "artifact_mismatch"
    assert exc_info.value.details["missing_count"] == 1


def test_loaded_artifacts_reject_invalid_relation_references() -> None:
    with pytest.raises(ApiStartupError) as exc_info:
        validate_loaded_artifacts(
            tables={"table_1": {}},
            search_documents={"table_1": {}},
            terms={"term_1": {}},
            term_outputs={"term_1": {}},
            table_terms={},
            clusters=[{"cluster_id": "cluster_1"}],
            cluster_by_term={},
            relations=[
                {
                    "relation_id": "relation_1",
                    "source_term_id": "term_1",
                    "target_term_id": "missing",
                    "relation_type": "related_to",
                }
            ],
        )

    assert exc_info.value.code == "artifact_mismatch"
    assert exc_info.value.details["invalid_count"] == 1
