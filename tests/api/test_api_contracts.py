from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from api.app.main import create_app
from statvocab.search.explain import NOTICE


class FakeState:
    config = None
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
                "knowledge_graph_loaded": True,
                "evaluation_loaded": True,
            },
            "artifact_counts": {
                "tables": 2,
                "search_documents": 2,
                "terms": 1,
                "clusters": 1,
                "relations": 1,
                "graph_nodes": 2,
                "graph_edges": 1,
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

    def term_list(
        self,
        *,
        page: int,
        page_size: int,
        category: str | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        rows = [
            {
                "term_id": "term_1",
                "canonical_term": "Employment",
                "category": "measure",
                "confidence": 0.9,
                "table_count": 1,
                "occurrence_count": 2,
                "cluster": {"cluster_id": "cluster_1", "domain": "labour market"},
                "relation_counts": {"incoming": 0, "outgoing": 1, "total": 1},
            },
            {
                "term_id": "term_2",
                "canonical_term": "Italy",
                "category": "dimension_value",
                "confidence": 0.8,
                "table_count": 1,
                "occurrence_count": 1,
                "cluster": None,
                "relation_counts": {"incoming": 0, "outgoing": 0, "total": 0},
            },
        ]
        if category:
            rows = [row for row in rows if row["category"] == category]
        if q:
            rows = [row for row in rows if q.casefold() in row["canonical_term"].casefold()]
        offset = (page - 1) * page_size
        return {
            "pagination": {"page": page, "page_size": page_size, "total": len(rows)},
            "items": rows[offset : offset + page_size],
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

    def graph_summary(self) -> dict[str, Any]:
        return {
            "run_id": "run_graph",
            "node_count": 2,
            "edge_count": 1,
            "node_type_counts": {"table": 1, "term": 1},
            "edge_type_counts": {"table_contains_term": 1},
        }

    def graph_focus_options(
        self,
        *,
        q: str,
        focus_type: str | None = None,
        page_size: int = 10,
    ) -> dict[str, Any]:
        candidates = [
            {
                "node_id": "term_1",
                "node_type": "term",
                "label": "Employment",
                "score": 100.0 if q == "term_1" else 90.0,
                "metadata": {"category": "measure"},
            },
            {
                "node_id": "table_1",
                "node_type": "table",
                "label": "Employment table",
                "score": 70.0,
                "metadata": {},
            },
        ]
        if len(q.strip()) < 2:
            candidates = []
        if focus_type:
            candidates = [item for item in candidates if item["node_type"] == focus_type]
        candidates.sort(key=lambda item: -float(item["score"]))
        return {"query": q.strip(), "focus_type": focus_type, "items": candidates[:page_size]}

    def graph_view(
        self,
        *,
        focus_type: str,
        focus_id: str,
        depth: int,
        min_weight: float,
        edge_type: str | None = None,
    ) -> dict[str, Any] | None:
        _ = edge_type
        if focus_type != "term" or focus_id != "term_1":
            return None
        return {
            "focus": {"focus_type": focus_type, "focus_id": focus_id},
            "depth": depth,
            "min_weight": min_weight,
            "edge_types": [],
            "nodes": [
                {
                    "node_id": "term_1",
                    "node_type": "term",
                    "label": "Employment",
                    "properties": {"category": "measure"},
                },
                {
                    "node_id": "table_1",
                    "node_type": "table",
                    "label": "Employment",
                    "properties": {"title": "Employment"},
                },
            ],
            "edges": [
                {
                    "edge_id": "graph_1",
                    "source_id": "table_1",
                    "target_id": "term_1",
                    "edge_type": "table_contains_term",
                    "weight": 1.0,
                    "directed": True,
                    "derived": False,
                    "evidence_ids": [],
                    "properties": {},
                }
            ],
            "limits": {"max_nodes": 250, "max_edges": 600},
            "total_available": {"nodes": 2, "edges": 1},
        }


def _client() -> TestClient:
    app = create_app(load_on_startup=False)
    state = FakeState()
    if state.config is None:
        from statvocab.config import load_config

        state.config = load_config("configs/evaluation.yaml")
    app.state.api_state = state
    return TestClient(app)


def test_openapi_documents_milestone_7_endpoints() -> None:
    client = _client()

    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/search" in paths
    assert "/api/tables/{table_id}" in paths
    assert "/api/terms" in paths
    assert "/api/terms/{term_id}" in paths
    assert "/api/clusters" in paths
    assert "/api/relations" in paths
    assert "/api/graph" in paths
    assert "/api/graph/focus-options" in paths
    assert "/api/graph/summary" in paths
    assert "/api/evaluation" in paths
    assert "/api/review/task" in paths
    assert "/api/review/answer" in paths
    assert "/api/review/stats" in paths
    assert "/api/review/compile" in paths
    assert "/api/health" in paths


def test_review_endpoints_use_human_loop_service(monkeypatch) -> None:
    import api.app.routes.review as review

    client = _client()

    monkeypatch.setattr(
        review,
        "next_review_task",
        lambda config, *, mode="all": {
            "task_id": "task_1",
            "task_type": "term_classification",
            "question": "What is this term?",
            "term_id": "term_1",
            "canonical_term": "Employment",
            "paired_term_id": "",
            "paired_canonical_term": "",
            "choices": [{"value": "measure", "label": "Measure", "shortcut": "1"}],
            "context": {},
            "metadata": {},
            "priority": 1,
            "hidden_qc": False,
        },
    )
    monkeypatch.setattr(
        review,
        "append_review_answer",
        lambda config, **kwargs: {"event": {"task_id": kwargs["task_id"]}, "stats": {}},
    )
    monkeypatch.setattr(
        review,
        "review_stats",
        lambda config: {
            "task_count": 1,
            "answered_task_count": 0,
            "remaining_task_count": 1,
            "event_count": 0,
            "task_type_counts": {"term_classification": 1},
            "answer_counts": {},
            "reviewer_counts": {},
            "tasks_path": "data/review/human_loop_tasks.jsonl",
            "events_path": "data/review/human_loop_events.jsonl",
        },
    )
    monkeypatch.setattr(
        review,
        "compile_human_labels",
        lambda config: (
            "report/human_loop_metrics.json",
            {
                "compiled_at": "2026-06-16T00:00:00+00:00",
                "events_count": 1,
                "compiled_classification_label_count": 1,
                "compiled_relation_label_count": 0,
                "classification_label_splits": {"train_dev": 1},
                "relation_label_splits": {},
                "agreement": {},
                "outputs": {},
                "classification_metrics_summary": {},
            },
        ),
    )

    assert client.get("/api/review/task").json()["task_id"] == "task_1"
    assert client.post(
        "/api/review/answer",
        json={"task_id": "task_1", "answer": "measure"},
    ).json()["event"]["task_id"] == "task_1"
    assert client.get("/api/review/stats").json()["task_count"] == 1
    assert client.post("/api/review/compile").json()[
        "compiled_classification_label_count"
    ] == 1


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
    assert payload["readiness"]["knowledge_graph_loaded"] is True


def test_collection_endpoints_use_page_and_page_size() -> None:
    client = _client()

    clusters = client.get("/api/clusters?page=1&page_size=5").json()
    relations = client.get("/api/relations?page=1&page_size=5&term_id=term_1").json()

    assert clusters["pagination"] == {"page": 1, "page_size": 5, "total": 1}
    assert relations["pagination"] == {"page": 1, "page_size": 5, "total": 1}


def test_terms_endpoint_paginates_and_filters() -> None:
    client = _client()

    payload = client.get("/api/terms?page=1&page_size=1&category=measure&q=employment").json()

    assert payload["pagination"] == {"page": 1, "page_size": 1, "total": 1}
    assert payload["items"][0]["term_id"] == "term_1"
    assert payload["items"][0]["relation_counts"]["total"] == 1


def test_evaluation_reports_available_and_missing_summaries() -> None:
    client = _client()

    payload = client.get("/api/evaluation").json()

    assert payload["summaries"]["retrieval"]["status"] == "available"
    assert payload["summaries"]["classification"]["status"] == "missing"


def test_graph_summary_and_focused_view_are_served() -> None:
    client = _client()

    summary = client.get("/api/graph/summary").json()
    graph = client.get("/api/graph?focus_type=term&focus_id=term_1&depth=1").json()

    assert summary["run_id"] == "run_graph"
    assert graph["focus"] == {"focus_type": "term", "focus_id": "term_1"}
    assert graph["nodes"][0]["node_id"] == "term_1"


def test_graph_focus_options_searches_focus_nodes() -> None:
    client = _client()

    payload = client.get("/api/graph/focus-options?q=employment&page_size=10").json()
    exact = client.get("/api/graph/focus-options?q=term_1&focus_type=term").json()
    empty = client.get("/api/graph/focus-options?q=e").json()

    assert payload["items"][0]["node_id"] == "term_1"
    assert {item["node_type"] for item in payload["items"]} == {"term", "table"}
    assert exact["items"][0]["node_id"] == "term_1"
    assert empty["items"] == []


def test_unknown_graph_focus_returns_stable_404() -> None:
    client = _client()

    response = client.get("/api/graph?focus_type=term&focus_id=missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "graph_focus_not_found"


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


def test_invalid_term_category_uses_stable_validation_error() -> None:
    client = _client()

    response = client.get("/api/terms?category=not_a_category")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_unbounded_search_pages_are_rejected() -> None:
    client = _client()

    response = client.post("/api/search", json={"query": "employment", "page": 1001})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
