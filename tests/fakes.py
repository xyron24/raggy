"""Shared test fakes: deterministic, no network, no keys."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, TypeVar

from pydantic import BaseModel
from qdrant_client import QdrantClient

from app.schemas import PassageScore, QueryKind, RerankVerdict, RouteDecision
from app.services.chunking import ChunkingService
from app.services.ingest import IngestService
from app.services.vector_store import QdrantStore

T = TypeVar("T", bound=BaseModel)


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


class FakeChatService:
    """Drop-in for ChatService with canned replies, decisions, and verdicts."""

    def __init__(
        self,
        reply: str = "Refunds are issued within 30 days [1].",
        decisions: list[RouteDecision] | None = None,
        verdicts: list[RerankVerdict] | None = None,
    ) -> None:
        self.reply = reply
        self._decisions = list(decisions or [])
        self._verdicts = list(verdicts or [])
        self.default_decision = RouteDecision(
            kind=QueryKind.FACTUAL, reason="default test route"
        )
        # Generous pass-through: out-of-range indices are ignored by the
        # re-ranker, so tests only script verdicts when filtering matters.
        self.default_verdict = RerankVerdict(
            scores=[
                PassageScore(index=i, score=1.0, reason="default test verdict")
                for i in range(1, 21)
            ]
        )
        self.complete_calls: list[tuple[str, str]] = []
        self.structured_calls: list[tuple[str, str, Any]] = []

    def complete(self, system: str, user: str) -> str:
        self.complete_calls.append((system, user))
        return self.reply

    def complete_structured(
        self, system: str, user: str, response_model: type[T]
    ) -> T:
        self.structured_calls.append((system, user, response_model))
        if getattr(response_model, "__name__", "") == "RerankVerdict":
            if self._verdicts:
                return self._verdicts.pop(0)  # type: ignore[return-value]
            return self.default_verdict  # type: ignore[return-value]
        if self._decisions:
            return self._decisions.pop(0)  # type: ignore[return-value]
        return self.default_decision  # type: ignore[return-value]


class FakeChatCompletions:
    """Mimics client.chat.completions with scripted create/parse replies."""

    def __init__(
        self,
        replies: list[str] | None = None,
        parsed_replies: list[BaseModel] | None = None,
    ) -> None:
        self._replies = list(replies or [])
        self._parsed = list(parsed_replies or [])
        self.create_calls: list[dict] = []
        self.parse_calls: list[dict] = []

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        text = self._replies.pop(0) if self._replies else ""
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        parsed = self._parsed.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
        )


class FakeChatClient:
    """Mimics the OpenAI client surface ChatService uses."""

    def __init__(
        self,
        replies: list[str] | None = None,
        parsed_replies: list[BaseModel] | None = None,
    ) -> None:
        self.chat = SimpleNamespace(
            completions=FakeChatCompletions(replies, parsed_replies)
        )


class ScriptedEmbeddingService:
    """Embedder returning fixed vectors per text, for retrieval comparisons."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]

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
