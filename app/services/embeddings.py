"""OpenAI embedding service with batching."""

from collections.abc import Sequence
from typing import Any

import openai


class EmbeddingService:
    """Embeds texts via the OpenAI embeddings API.

    Rate-limit retries already come from the OpenAI SDK (exponential
    backoff, enabled by default); this class adds batching, input
    validation, and a swappable client for tests.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.batch_size = batch_size
        self._client = client or openai.OpenAI(api_key=api_key)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many texts, sending at most `batch_size` per API call."""
        if not texts:
            return []
        for text in texts:
            if not text or not text.strip():
                raise ValueError("cannot embed blank text")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self._client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        return self.embed_texts([query])[0]
