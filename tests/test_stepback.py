"""Step-Back tests: broader questions and merged retrieval."""

import pytest
from qdrant_client import QdrantClient

from app.schemas import Chunk
from app.services.hyde import HyDEService
from app.services.retriever import RetrieverService
from app.services.stepback import STEPBACK_SYSTEM_PROMPT, StepBackService
from app.services.vector_store import QdrantStore
from tests.fakes import FakeChatService, FakeEmbeddingService, ScriptedEmbeddingService


def make_store(dim: int = 4) -> QdrantStore:
    store = QdrantStore(
        url=":memory:",
        collection="test_docs",
        vector_size=dim,
        client=QdrantClient(":memory:"),
    )
    store.ensure_collection()
    return store


def make_retriever(embeddings, store, reply: str):
    chat = FakeChatService(reply=reply)
    retriever = RetrieverService(
        embeddings, store, HyDEService(chat), StepBackService(chat)
    )
    return retriever, chat


def test_step_back_returns_stripped_question():
    chat = FakeChatService(reply="  broader q?  ")
    assert StepBackService(chat).step_back("specific q?") == "broader q?"
    system, user = chat.complete_calls[0]
    assert system == STEPBACK_SYSTEM_PROMPT
    assert user == "specific q?"


def test_retrieve_with_stepback_merges_both_sides():
    store = make_store()
    store.upsert(
        [
            Chunk(chunk_id="a:0", doc_id="a", position=0, text="alpha doc"),
            Chunk(chunk_id="b:0", doc_id="b", position=0, text="beta doc"),
        ],
        [[1, 0, 0, 0], [0, 1, 0, 0]],
    )
    embeddings = ScriptedEmbeddingService(
        {"specific q": [0, 1, 0, 0], "broad q": [1, 0, 0, 0]}
    )
    retriever, _ = make_retriever(embeddings, store, "broad q")
    hits = retriever.retrieve_with_stepback("specific q", top_k=2)
    assert {h.chunk.chunk_id for h in hits} == {"a:0", "b:0"}
    assert embeddings.calls == [["specific q"], ["broad q"]]


def test_retrieve_with_stepback_keeps_best_score():
    store = make_store()
    store.upsert(
        [Chunk(chunk_id="c:0", doc_id="c", position=0, text="shared doc")],
        [[1, 0.5, 0, 0]],
    )
    embeddings = ScriptedEmbeddingService(
        {"specific q": [1, 0, 0, 0], "broad q": [0, 1, 0, 0]}
    )
    retriever, _ = make_retriever(embeddings, store, "broad q")
    hits = retriever.retrieve_with_stepback("specific q", top_k=2)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(0.8944, abs=1e-3)


def test_identical_stepback_skips_second_search():
    store = make_store()
    embeddings = FakeEmbeddingService(dim=4)
    retriever, _ = make_retriever(embeddings, store, "same q")
    retriever.retrieve_with_stepback("same q", top_k=1)
    assert embeddings.calls == [["same q"]]


def test_blank_stepback_skips_second_search():
    store = make_store()
    embeddings = FakeEmbeddingService(dim=4)
    retriever, _ = make_retriever(embeddings, store, "   ")
    retriever.retrieve_with_stepback("specific q", top_k=1)
    assert embeddings.calls == [["specific q"]]


def test_retrieve_with_stepback_skips_llm_when_store_empty():
    store = QdrantStore(
        url=":memory:",
        collection="missing",
        vector_size=4,
        client=QdrantClient(":memory:"),
    )
    retriever, chat = make_retriever(FakeEmbeddingService(dim=4), store, "broad q")
    assert retriever.retrieve_with_stepback("q?", top_k=1) == []
    assert chat.complete_calls == []
