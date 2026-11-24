"""Vector store tests against in-memory Qdrant (no server needed)."""

import pytest
from qdrant_client import QdrantClient

from app.schemas import Chunk
from app.services.vector_store import QdrantStore, point_id_for


def make_chunk(i: int) -> Chunk:
    return Chunk(chunk_id=f"d:{i}", doc_id="d", position=i, text=f"passage {i}")


@pytest.fixture
def store() -> QdrantStore:
    client = QdrantClient(":memory:")
    store = QdrantStore(
        url=":memory:", collection="test_docs", vector_size=4, client=client
    )
    store.ensure_collection()
    return store


def test_point_ids_are_deterministic():
    assert point_id_for("d:0") == point_id_for("d:0")
    assert point_id_for("d:0") != point_id_for("d:1")


def test_ensure_collection_is_idempotent(store: QdrantStore):
    store.ensure_collection()
    assert store.count() == 0


def test_count_missing_collection_is_zero():
    store = QdrantStore(
        url=":memory:",
        collection="missing",
        vector_size=4,
        client=QdrantClient(":memory:"),
    )
    assert store.count() == 0


def test_exists_reports_collection_presence():
    store = QdrantStore(
        url=":memory:",
        collection="fresh",
        vector_size=4,
        client=QdrantClient(":memory:"),
    )
    assert store.exists() is False
    store.ensure_collection()
    assert store.exists() is True


def test_upsert_and_search_roundtrip(store: QdrantStore):
    chunks = [make_chunk(0), make_chunk(1)]
    assert store.upsert(chunks, [[1, 0, 0, 0], [0, 1, 0, 0]]) == 2
    assert store.count() == 2
    hits = store.search([1, 0, 0, 0], top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "d:0"
    assert hits[0].chunk.text == "passage 0"
    assert hits[0].score == pytest.approx(1.0)


def test_upsert_same_id_overwrites(store: QdrantStore):
    store.upsert([make_chunk(0)], [[1, 0, 0, 0]])
    store.upsert([make_chunk(0)], [[0, 0, 1, 0]])
    assert store.count() == 1
    hits = store.search([0, 0, 1, 0], top_k=1)
    assert hits[0].chunk.chunk_id == "d:0"


def test_upsert_misaligned_inputs_rejected(store: QdrantStore):
    with pytest.raises(ValueError):
        store.upsert([make_chunk(0)], [])


def test_upsert_empty_is_noop(store: QdrantStore):
    assert store.upsert([], []) == 0


def test_search_rejects_bad_top_k(store: QdrantStore):
    with pytest.raises(ValueError):
        store.search([1, 0, 0, 0], top_k=0)


def test_collection_width_mismatch_raises():
    client = QdrantClient(":memory:")
    QdrantStore(url=":memory:", collection="c", vector_size=4, client=client).ensure_collection()
    with pytest.raises(ValueError, match="width"):
        QdrantStore(url=":memory:", collection="c", vector_size=8, client=client).ensure_collection()
