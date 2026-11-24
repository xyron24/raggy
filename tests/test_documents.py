"""Document API tests with overridden dependencies (no keys, no servers)."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_ingest_service, get_settings, get_vector_store
from app.main import app
from tests.fakes import make_test_ingest_service


@pytest.fixture
def client() -> Iterator[TestClient]:
    # One shared service: in-memory Qdrant keeps state inside the client,
    # so each request must reuse it (a real server holds state itself).
    # Stats shares the same store, like one collection server-side.
    service = make_test_ingest_service()
    app.dependency_overrides[get_ingest_service] = lambda: service
    app.dependency_overrides[get_vector_store] = lambda: service.store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_ingest_happy_path(client: TestClient):
    response = client.post(
        "/documents/ingest",
        json={"doc_id": "faq1", "title": "FAQ", "text": "Refunds are issued within 30 days of purchase."},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["doc_id"] == "faq1"
    assert body["num_chunks"] == 1
    assert body["chunk_ids"] == ["faq1:0"]

    stats = client.get("/documents/stats").json()
    assert stats["count"] == 1
    assert stats["collection"] == "test_docs"
    assert stats["vector_size"] == 8


def test_ingest_generates_doc_id_when_missing(client: TestClient):
    response = client.post("/documents/ingest", json={"text": "hello world"})
    assert response.status_code == 201
    body = response.json()
    assert body["doc_id"]
    assert body["chunk_ids"] == [f"{body['doc_id']}:0"]


def test_reingest_same_id_overwrites(client: TestClient):
    client.post("/documents/ingest", json={"doc_id": "d", "text": "version one"})
    client.post("/documents/ingest", json={"doc_id": "d", "text": "version two"})
    assert client.get("/documents/stats").json()["count"] == 1


def test_ingest_rejects_empty_text(client: TestClient):
    response = client.post("/documents/ingest", json={"text": ""})
    assert response.status_code == 422


def test_ingest_without_api_key_returns_503():
    app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="")
    try:
        response = TestClient(app).post("/documents/ingest", json={"text": "hi"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_stats_works_without_api_key():
    service = make_test_ingest_service()
    app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="")
    app.dependency_overrides[get_vector_store] = lambda: service.store
    try:
        response = TestClient(app).get("/documents/stats")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["count"] == 0
