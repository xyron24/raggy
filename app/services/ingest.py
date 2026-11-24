"""Document ingest orchestration: split, embed, store."""

from uuid import uuid4

from app.schemas import DocumentIn, IngestResponse
from app.services.chunking import ChunkingService
from app.services.embeddings import EmbeddingService
from app.services.vector_store import QdrantStore


class IngestService:
    """Coordinates the ingest pipeline across the three stage services."""

    def __init__(
        self,
        chunking: ChunkingService,
        embeddings: EmbeddingService,
        store: QdrantStore,
    ) -> None:
        self._chunking = chunking
        self._embeddings = embeddings
        self._store = store

    def ingest(self, document: DocumentIn) -> IngestResponse:
        """Split, embed, and upsert one document.

        The collection check runs before any embedding call so a dead
        vector DB fails fast, ahead of the billed API calls.
        """
        doc_id = document.doc_id or uuid4().hex
        resolved = document.model_copy(update={"doc_id": doc_id})
        self._store.ensure_collection()
        chunks = self._chunking.chunk_document(resolved)
        vectors = self._embeddings.embed_texts([c.text for c in chunks])
        self._store.upsert(chunks, vectors)
        return IngestResponse(
            doc_id=doc_id,
            num_chunks=len(chunks),
            chunk_ids=[c.chunk_id for c in chunks],
        )

    @property
    def store(self) -> QdrantStore:
        """The underlying vector store (shared with stats endpoints)."""
        return self._store
