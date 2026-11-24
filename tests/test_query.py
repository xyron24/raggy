"""Query API tests: grounded answers, citations, edge cases."""

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.config import Settings
from app.dependencies import get_answer_service, get_settings
from app.main import app
from app.schemas import DocumentIn, QueryKind, RouteDecision
from app.services.answer import AnswerService
from app.services.chunking import ChunkingService
from app.services.hyde import HyDEService
from app.services.ingest import IngestService
from app.services.retriever import RetrieverService
from app.services.router import QueryRouter
from app.services.stepback import StepBackService
from app.services.vector_store import QdrantStore
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
    answer = AnswerService(
        RetrieverService(
            FakeEmbeddingService(dim), store, HyDEService(chat), StepBackService(chat)
        ),
        chat,
        QueryRouter(chat),
        hyde_enabled=True,
        stepback_enabled=True,
    )
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
    assert body["route"] == "factual"
    assert body["confidence"] == pytest.approx(1.0)
    assert body["expanded"] is False


def test_prompt_contains_passages_and_question(harness: Harness):
    seed(harness)
    harness.client.post("/query", json={"query": "What is the refund policy?"})
    assert len(harness.chat.complete_calls) == 1
    system, user = harness.chat.complete_calls[0]
    assert "[n]" in system
    assert "Refunds are issued within 30 days" in user
    assert "What is the refund policy?" in user


def test_empty_store_skips_synthesis_call(harness: Harness):
    response = harness.client.post("/query", json={"query": "anything?"})
    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert "No relevant passages" in body["answer"]
    assert body["route"] == "factual"
    assert body["confidence"] == 0.0
    assert body["expanded"] is False
    assert harness.chat.complete_calls == []
    assert len(harness.chat.structured_calls) == 1


def test_top_k_limits_sources(harness: Harness):
    for i in range(3):
        harness.ingest.ingest(DocumentIn(doc_id=f"d{i}", text=f"fact number {i} here"))
    body = harness.client.post("/query", json={"query": "facts?", "top_k": 1}).json()
    assert len(body["sources"]) == 1
    assert body["sources"][0]["index"] == 1


def test_response_route_follows_router_decision(harness: Harness):
    seed(harness)
    harness.chat.default_decision = RouteDecision(
        kind=QueryKind.COMPLEX, reason="seeded decision"
    )
    body = harness.client.post("/query", json={"query": "compare everything"}).json()
    assert body["route"] == "complex"
    assert body["expanded"] is True
    assert len(body["sources"]) == 1
    system, user, model = harness.chat.structured_calls[0]
    assert "compare everything" in user
    assert model.__name__ == "RouteDecision"
    assert len(harness.chat.complete_calls) == 2
    stepback_system, stepback_user = harness.chat.complete_calls[0]
    assert stepback_user == "compare everything"
    assert "broader" in stepback_system


def test_vague_route_expands_via_hyde(harness: Harness):
    seed(harness)
    harness.chat.default_decision = RouteDecision(
        kind=QueryKind.VAGUE, reason="seeded decision"
    )
    body = harness.client.post("/query", json={"query": "refunds?"}).json()
    assert body["route"] == "vague"
    assert body["expanded"] is True
    assert len(harness.chat.complete_calls) == 2
    hyde_system, hyde_user = harness.chat.complete_calls[0]
    assert hyde_user == "refunds?"
    assert "excerpt" in hyde_system


def test_hyde_flag_off_keeps_plain_retrieval():
    dim = 8
    store = QdrantStore(
        url=":memory:",
        collection="test_docs",
        vector_size=dim,
        client=QdrantClient(":memory:"),
    )
    chat = FakeChatService(
        decisions=[RouteDecision(kind=QueryKind.VAGUE, reason="seeded")]
    )
    ingest = IngestService(ChunkingService(), FakeEmbeddingService(dim), store)
    ingest.ingest(DocumentIn(doc_id="d", text="some fact here"))
    answer = AnswerService(
        RetrieverService(
            FakeEmbeddingService(dim), store, HyDEService(chat), StepBackService(chat)
        ),
        chat,
        QueryRouter(chat),
        hyde_enabled=False,
    )
    app.dependency_overrides[get_answer_service] = lambda: answer
    try:
        body = TestClient(app).post("/query", json={"query": "vague?"}).json()
    finally:
        app.dependency_overrides.clear()
    assert body["route"] == "vague"
    assert body["expanded"] is False
    assert len(chat.complete_calls) == 1


def test_stepback_flag_off_keeps_plain_retrieval():
    dim = 8
    store = QdrantStore(
        url=":memory:",
        collection="test_docs",
        vector_size=dim,
        client=QdrantClient(":memory:"),
    )
    chat = FakeChatService(
        decisions=[RouteDecision(kind=QueryKind.COMPLEX, reason="seeded")]
    )
    ingest = IngestService(ChunkingService(), FakeEmbeddingService(dim), store)
    ingest.ingest(DocumentIn(doc_id="d", text="some fact here"))
    answer = AnswerService(
        RetrieverService(
            FakeEmbeddingService(dim), store, HyDEService(chat), StepBackService(chat)
        ),
        chat,
        QueryRouter(chat),
        stepback_enabled=False,
    )
    app.dependency_overrides[get_answer_service] = lambda: answer
    try:
        body = TestClient(app).post("/query", json={"query": "complex?"}).json()
    finally:
        app.dependency_overrides.clear()
    assert body["route"] == "complex"
    assert body["expanded"] is False
    assert len(chat.complete_calls) == 1


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
