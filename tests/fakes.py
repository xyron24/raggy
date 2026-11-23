"""Shared test fakes: deterministic, no network, no keys."""

from collections.abc import Sequence

from qdrant_client import QdrantClient

from app.services.chunking import ChunkingService
from app.services.ingest import IngestService
from app.services.vector_store import QdrantStore


class FakeEmbeddingService:
    """Drop-in for EmbeddingService returning deterministic vectors."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text))] * self.dim for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


def make_test_ingest_service(dim: int = 8) -> IngestService:
    """IngestService wired to real chunking, fake embeddings, memory Qdrant."""
    store = QdrantStore(
        url=":memory:",
        collection="test_docs",
        vector_size=dim,
        client=QdrantClient(":memory:"),
    )
    return IngestService(
        chunking=ChunkingService(),
        embeddings=FakeEmbeddingService(dim=dim),  # type: ignore[arg-type]
        store=store,
    )
