from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from api.app.main import create_app
from statvocab.search.explain import NOTICE


class FakeState:
    evaluation = {
        "retrieval": {"status": "available", "path": "report/retrieval_metrics.json", "data": {}},
        "classification": {
            "status": "missing",
            "path": "report/classification_metrics.json",
            "data": {},
        },
    }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "config_name": "core",
            "corpus": "core",
            "loaded_search_run_id": "run_search",
            "document_count": 2,
            "readiness": {
                "config_loaded": True,
                "search_index_ready": True,
                "tables_loaded": True,
                "terms_loaded": True,
                "clusters_loaded": True,
                "relations_loaded": True,
                "evaluation_loaded": True,
            },
            "artifact_counts": {
                "tables": 2,
                "search_documents": 2,
                "terms": 1,
                "clusters": 1,
                "relations": 1,
            },
            "warnings": [],
        }

    def search(self, query: str, *, page: int, page_size: int, system: str) -> dict[str, Any]:
        _ = query, system
        return {
            "query": {"original": "employment in Italy", "semantic_remainder": "employment"},
            "notice": NOTICE,
            "pagination": {"page": page, "page_size": page_size, "total": 1},
            "results": [{"rank": 1, "table_id": "table_1", "notice": NOTICE}],
        }

    def table_detail(self, table_id: str) -> dict[str, Any] | None:
        if table_id != "table_1":
            return None
        return {
            "table_id": table_id,
            "title": "Employment",
            "source_url": "https://example.test/table_1",
            "parse_status": "parsed",
            "metadata": {"row_count": 10},
            "vocabulary": {"measures": ["Employment"]},
            "geographies": [],
            "times": [],
            "evidence": [],
            "related_terms": [{"term_id": "term_1", "canonical_term": "Employment"}],
        }

    def term_detail(self, term_id: str) -> dict[str, Any] | None:
        if term_id != "term_1":
            return None
        return {
            "term_id": term_id,
            "canonical_term": "Employment",
            "category": "measure",
            "confidence": 0.9,
            "provenance": {"run_id": "run_classification"},
            "occurrence_ids": ["occ_1"],
            "table_appearances": [{"table_id": "table_1"}],
            "cluster": {"cluster_id": "cluster_1"},
            "relations": {"incoming": [], "outgoing": []},
        }

    def cluster_list(self, *, page: int, page_size: int) -> dict[str, Any]:
        return {
            "pagination": {"page": page, "page_size": page_size, "total": 1},
            "items": [{"cluster_id": "cluster_1", "size": 1}],
        }

    def relation_list(
        self,
        *,
        page: int,
        page_size: int,
        term_id: str | None = None,
        relation_type: str | None = None,
    ) -> dict[str, Any]:
        _ = term_id, relation_type
        return {
            "pagination": {"page": page, "page_size": page_size, "total": 1},
            "items": [{"relation_id": "relation_1", "relation_type": "related_to"}],
        }


def _client() -> TestClient:
    app = create_app(load_on_startup=False)
    app.state.api_state = FakeState()
    return TestClient(app)


def test_openapi_documents_milestone_7_endpoints() -> None:
    client = _client()

    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/search" in paths
    assert "/api/tables/{table_id}" in paths
    assert "/api/terms/{term_id}" in paths
    assert "/api/clusters" in paths
    assert "/api/relations" in paths
    assert "/api/evaluation" in paths
    assert "/api/health" in paths


def test_checked_in_openapi_matches_live_schema() -> None:
    client = _client()
    expected_path = Path("api/openapi.json")

    assert expected_path.exists()
    assert json.loads(expected_path.read_text(encoding="utf-8")) == client.get(
        "/openapi.json"
    ).json()


def test_search_returns_notice_and_pagination() -> None:
    client = _client()

    response = client.post(
        "/api/search",
        json={"query": "employment in Italy", "page": 1, "page_size": 10},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["notice"] == NOTICE
    assert payload["pagination"] == {"page": 1, "page_size": 10, "total": 1}
    assert payload["results"][0]["table_id"] == "table_1"
    assert payload["results"][0]["notice"] == NOTICE


def test_unknown_ids_return_stable_404_errors() -> None:
    client = _client()

    table_response = client.get("/api/tables/unknown")
    term_response = client.get("/api/terms/unknown")

    assert table_response.status_code == 404
    assert table_response.json()["error"]["code"] == "table_not_found"
    assert term_response.status_code == 404
    assert term_response.json()["error"]["code"] == "term_not_found"


def test_health_reports_loaded_run_and_index_readiness() -> None:
    client = _client()

    payload = client.get("/api/health").json()

    assert payload["loaded_search_run_id"] == "run_search"
    assert payload["readiness"]["search_index_ready"] is True
    assert payload["artifact_counts"]["tables"] == 2


def test_collection_endpoints_use_page_and_page_size() -> None:
    client = _client()

    clusters = client.get("/api/clusters?page=1&page_size=5").json()
    relations = client.get("/api/relations?page=1&page_size=5&term_id=term_1").json()

    assert clusters["pagination"] == {"page": 1, "page_size": 5, "total": 1}
    assert relations["pagination"] == {"page": 1, "page_size": 5, "total": 1}


def test_evaluation_reports_available_and_missing_summaries() -> None:
    client = _client()

    payload = client.get("/api/evaluation").json()

    assert payload["summaries"]["retrieval"]["status"] == "available"
    assert payload["summaries"]["classification"]["status"] == "missing"


def test_validation_errors_use_stable_error_shape() -> None:
    client = _client()

    response = client.post("/api/search", json={"query": "", "page": 0})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_invalid_relation_type_uses_stable_validation_error() -> None:
    client = _client()

    response = client.get("/api/relations?relation_type=not_a_relation")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_unbounded_search_pages_are_rejected() -> None:
    client = _client()

    response = client.post("/api/search", json={"query": "employment", "page": 1001})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
