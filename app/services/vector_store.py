"""Qdrant-backed vector storage for chunks."""

from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.schemas import Chunk, SearchResult


def point_id_for(chunk_id: str) -> str:
    """Deterministic Qdrant point id derived from our chunk id.

    Qdrant point ids must be ints or UUIDs, while ours look like
    "doc:3", so we derive a stable UUIDv5. Same chunk always maps to
    the same point, which keeps re-ingests idempotent.
    """
    return str(uuid5(NAMESPACE_URL, f"raggy:{chunk_id}"))


class QdrantStore:
    """Thin wrapper around the Qdrant client for chunk vectors."""

    def __init__(
        self,
        url: str,
        collection: str,
        vector_size: int,
        client: QdrantClient | None = None,
    ) -> None:
        self._client = client or QdrantClient(url=url)
        self.collection = collection
        self.vector_size = vector_size

    def exists(self) -> bool:
        """True when the collection has been created."""
        return self._client.collection_exists(self.collection)

    def ensure_collection(self) -> None:
        """Create the collection if missing; verify width if present."""
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )
            return
        actual = self._client.get_collection(self.collection).config.params.vectors.size  # type: ignore[union-attr]
        if actual != self.vector_size:
            raise ValueError(
                f"collection '{self.collection}' has width {actual}, "
                f"expected {self.vector_size} (embedding model changed?)"
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Store chunk vectors, overwriting any with the same ids."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must align one-to-one")
        if not chunks:
            return 0
        points = [
            PointStruct(
                id=point_id_for(chunk.chunk_id),
                vector=vector,
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return the top-k most similar chunks for a query vector."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        hits = self._client.query_points(
            collection_name=self.collection, query=vector, limit=top_k
        ).points
        return [
            SearchResult(chunk=Chunk(**hit.payload), score=hit.score)  # type: ignore[arg-type]
            for hit in hits
        ]

    def count(self) -> int:
        """Number of points stored (0 when the collection is missing)."""
        if not self.exists():
            return 0
        return self._client.count(collection_name=self.collection).count
