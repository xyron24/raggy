"""Embedding service tests using a fake OpenAI client (no network)."""

from types import SimpleNamespace

import pytest

from app.services.embeddings import EmbeddingService


class FakeEmbeddingsEndpoint:
    """Stands in for client.embeddings; records batches, returns canned vectors."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.batches: list[list[str]] = []

    def create(self, model: str, input: list[str]):
        assert model == "test-model"
        self.batches.append(list(input))
        data = [
            SimpleNamespace(embedding=[float(len(text))] * self.dim)
            for text in input
        ]
        return SimpleNamespace(data=data)


class FakeOpenAIClient:
    def __init__(self, dim: int = 8) -> None:
        self.embeddings = FakeEmbeddingsEndpoint(dim)


def make_service(batch_size: int = 100, dim: int = 8) -> tuple[EmbeddingService, FakeOpenAIClient]:
    client = FakeOpenAIClient(dim)
    service = EmbeddingService(
        api_key="test-key", model="test-model", batch_size=batch_size, client=client
    )
    return service, client


def test_embed_texts_batches_requests():
    service, client = make_service(batch_size=2)
    vectors = service.embed_texts(["a", "bb", "ccc", "dddd", "eeeee"])
    assert [len(batch) for batch in client.embeddings.batches] == [2, 2, 1]
    assert len(vectors) == 5
    assert vectors[2] == [3.0] * 8


def test_embed_texts_empty_input_makes_no_calls():
    service, client = make_service()
    assert service.embed_texts([]) == []
    assert client.embeddings.batches == []


def test_embed_texts_rejects_blank_input():
    service, _ = make_service()
    with pytest.raises(ValueError):
        service.embed_texts(["ok", "   "])


def test_embed_query_returns_single_vector():
    service, _ = make_service()
    assert service.embed_query("hello") == [5.0] * 8


def test_missing_api_key_rejected():
    with pytest.raises(ValueError):
        EmbeddingService(api_key="", client=FakeOpenAIClient())


def test_invalid_batch_size_rejected():
    with pytest.raises(ValueError):
        EmbeddingService(api_key="k", batch_size=0, client=FakeOpenAIClient())
