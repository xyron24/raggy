"""Query-time retrieval: embed the question, search the vector store."""

from app.schemas import SearchResult
from app.services.embeddings import EmbeddingService
from app.services.hyde import HyDEService
from app.services.vector_store import QdrantStore


class RetrieverService:
    """Turns a question into ranked candidate passages."""

    def __init__(
        self,
        embeddings: EmbeddingService,
        store: QdrantStore,
        hyde: HyDEService,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._hyde = hyde

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
