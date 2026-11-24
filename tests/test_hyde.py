"""HyDE tests: hypothetical generation and expansion-driven retrieval."""

from qdrant_client import QdrantClient

from app.schemas import Chunk
from app.services.hyde import HYDE_SYSTEM_PROMPT, HyDEService
from app.services.retriever import RetrieverService
from app.services.stepback import StepBackService
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


def test_imagine_returns_stripped_passage():
    chat = FakeChatService(reply="  imagined passage  ")
    assert HyDEService(chat).imagine("short q?") == "imagined passage"
    system, user = chat.complete_calls[0]
    assert system == HYDE_SYSTEM_PROMPT
    assert user == "short q?"


def test_retrieve_with_hyde_embeds_hypothetical_not_query():
    store = make_store()
    store.upsert(
        [Chunk(chunk_id="d:0", doc_id="d", position=0, text="doc text")],
        [[1, 0, 0, 0]],
    )
    embeddings = FakeEmbeddingService(dim=4)
    chat = FakeChatService(reply="imagined passage")
    retriever = RetrieverService(
        embeddings, store, HyDEService(chat), StepBackService(chat)
    )
    hits = retriever.retrieve_with_hyde("short q?", top_k=1)
    assert embeddings.calls == [["imagined passage"]]
    assert [h.chunk.chunk_id for h in hits] == ["d:0"]


def test_retrieve_with_hyde_falls_back_to_query_on_blank():
    store = make_store()
    embeddings = FakeEmbeddingService(dim=4)
    chat = FakeChatService(reply="   ")
    retriever = RetrieverService(
        embeddings, store, HyDEService(chat), StepBackService(chat)
    )
    retriever.retrieve_with_hyde("short q?", top_k=1)
    assert embeddings.calls == [["short q?"]]


def test_retrieve_with_hyde_skips_llm_when_store_empty():
    store = QdrantStore(
        url=":memory:",
        collection="missing",
        vector_size=4,
        client=QdrantClient(":memory:"),
    )
    chat = FakeChatService()
    retriever = RetrieverService(
        FakeEmbeddingService(dim=4), store, HyDEService(chat), StepBackService(chat)
    )
    assert retriever.retrieve_with_hyde("q?", top_k=1) == []
    assert chat.complete_calls == []


def test_hyde_retrieval_can_differ_from_plain():
    # Doc A matches the hypothetical's vector; doc B matches the raw query.
    # Plain retrieval surfaces B, HyDE surfaces A: the expansion changes
    # what gets retrieved, which is the whole point of HyDE.
    store = make_store()
    store.upsert(
        [
            Chunk(chunk_id="a:0", doc_id="a", position=0, text="alpha doc"),
            Chunk(chunk_id="b:0", doc_id="b", position=0, text="beta doc"),
        ],
        [[1, 0, 0, 0], [0, 1, 0, 0]],
    )
    embeddings = ScriptedEmbeddingService(
        {"raw query": [0, 1, 0, 0], "imagined answer": [1, 0, 0, 0]}
    )
    chat = FakeChatService(reply="imagined answer")
    retriever = RetrieverService(
        embeddings,  # type: ignore[arg-type]
        store,
        HyDEService(chat),
        StepBackService(chat),
    )
    plain = retriever.retrieve("raw query", top_k=1)
    expanded = retriever.retrieve_with_hyde("raw query", top_k=1)
    assert [h.chunk.chunk_id for h in plain] == ["b:0"]
    assert [h.chunk.chunk_id for h in expanded] == ["a:0"]
