"""Query-time retrieval: embed the question, search the vector store."""

from app.schemas import SearchResult
from app.services.embeddings import EmbeddingService
from app.services.hyde import HyDEService
from app.services.stepback import StepBackService
from app.services.vector_store import QdrantStore


class RetrieverService:
    """Turns a question into ranked candidate passages."""

    def __init__(
        self,
        embeddings: EmbeddingService,
        store: QdrantStore,
        hyde: HyDEService,
        stepback: StepBackService,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._hyde = hyde
        self._stepback = stepback

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return up to top_k passages for the query.

        A missing collection means nothing was ever ingested, which is an
        empty result, not an error.
        """
        if not self._store.exists():
            return []
        vector = self._embeddings.embed_query(query)
        return self._store.search(vector, top_k=top_k)

    def retrieve_with_hyde(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve by embedding an imagined answer instead of the query.

        Falls back to the raw query when imagination returns nothing, so a
        degenerate generation degrades to plain retrieval instead of failing.
        """
        if not self._store.exists():
            return []
        hypothetical = self._hyde.imagine(query)
        vector = self._embeddings.embed_query(hypothetical or query)
        return self._store.search(vector, top_k=top_k)

    def retrieve_with_stepback(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve for the query and its broader step-back, merged.

        Both sides fetch top_k; the union is deduped by chunk (keeping the
        best score), re-sorted, and cut to top_k. A blank or identical
        step-back question skips the second search.
        """
        if not self._store.exists():
            return []
        broad = self._stepback.step_back(query)
        if not broad or broad == query:
            return self.retrieve(query, top_k=top_k)
        merged: dict[str, SearchResult] = {}
        for question in (query, broad):
            for hit in self.retrieve(question, top_k=top_k):
                current = merged.get(hit.chunk.chunk_id)
                if current is None or hit.score > current.score:
                    merged[hit.chunk.chunk_id] = hit
        return sorted(merged.values(), key=lambda h: h.score, reverse=True)[:top_k]
