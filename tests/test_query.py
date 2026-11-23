"""Query API tests: grounded answers, citations, edge cases."""

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.config import Settings
from app.dependencies import get_answer_service, get_settings
from app.main import app
from app.services.answer import AnswerService
from app.services.chunking import ChunkingService
from app.services.ingest import IngestService
from app.services.retriever import RetrieverService
from app.services.vector_store import QdrantStore
from app.schemas import DocumentIn
from tests.fakes import FakeChatService, FakeEmbeddingService


@dataclass
class Harness:
    client: TestClient
    ingest: IngestService
    chat: FakeChatService


@pytest.fixture
def harness() -> Iterator[Harness]:
    # Ingest and answer share one in-memory store, like they share one
    # server-side collection in production.
    dim = 8
    store = QdrantStore(
        url=":memory:",
        collection="test_docs",
        vector_size=dim,
        client=QdrantClient(":memory:"),
    )
    chat = FakeChatService()
    ingest = IngestService(ChunkingService(), FakeEmbeddingService(dim), store)
    answer = AnswerService(RetrieverService(FakeEmbeddingService(dim), store), chat)
    app.dependency_overrides[get_answer_service] = lambda: answer
    yield Harness(client=TestClient(app), ingest=ingest, chat=chat)
    app.dependency_overrides.clear()


def seed(harness: Harness) -> None:
    harness.ingest.ingest(
        DocumentIn(
            doc_id="faq1",
            title="FAQ",
            text="Refunds are issued within 30 days of purchase.",
        )
    )


def test_query_happy_path_returns_cited_answer(harness: Harness):
    seed(harness)
    response = harness.client.post("/query", json={"query": "What is the refund policy?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Refunds are issued within 30 days [1]."
    assert len(body["sources"]) == 1
    source = body["sources"][0]
    assert source["index"] == 1
    assert source["chunk_id"] == "faq1:0"
    assert source["doc_id"] == "faq1"
    assert "Refunds are issued" in source["text"]


def test_prompt_contains_passages_and_question(harness: Harness):
    seed(harness)
    harness.client.post("/query", json={"query": "What is the refund policy?"})
    assert len(harness.chat.calls) == 1
    system, user = harness.chat.calls[0]
    assert "[n]" in system
    assert "Refunds are issued within 30 days" in user
    assert "What is the refund policy?" in user


def test_empty_store_answers_without_llm_call(harness: Harness):
    response = harness.client.post("/query", json={"query": "anything?"})
    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert "No relevant passages" in body["answer"]
    assert harness.chat.calls == []


def test_top_k_limits_sources(harness: Harness):
    for i in range(3):
        harness.ingest.ingest(DocumentIn(doc_id=f"d{i}", text=f"fact number {i} here"))
    body = harness.client.post("/query", json={"query": "facts?", "top_k": 1}).json()
    assert len(body["sources"]) == 1
    assert body["sources"][0]["index"] == 1


def test_query_rejects_bad_input(harness: Harness):
    assert harness.client.post("/query", json={"query": ""}).status_code == 422
    assert harness.client.post("/query", json={"query": "q", "top_k": 0}).status_code == 422
    assert harness.client.post("/query", json={"query": "q", "top_k": 99}).status_code == 422


def test_query_without_api_key_returns_503():
    app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="")
    try:
        response = TestClient(app).post("/query", json={"query": "hi"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
