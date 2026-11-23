"""Chunking service tests: splitting, overlap, metadata, edge cases."""

import pytest
from pydantic import ValidationError

from app.schemas import DocumentIn
from app.services.chunking import ChunkingService


def test_short_text_yields_single_chunk():
    service = ChunkingService(chunk_size=500, chunk_overlap=50)
    chunks = service.chunk_document(DocumentIn(doc_id="d1", text="Hello world"))
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d1:0"
    assert chunks[0].position == 0
    assert chunks[0].text == "Hello world"


def test_long_text_splits_with_overlap():
    service = ChunkingService(chunk_size=100, chunk_overlap=20)
    text = " ".join(f"word{i}" for i in range(200))
    chunks = service.chunk_document(DocumentIn(doc_id="d2", text=text))
    assert len(chunks) > 1
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == "d2:0"
    # Overlap: neighboring chunks share some words.
    assert any(
        set(a.text.split()) & set(b.text.split())
        for a, b in zip(chunks, chunks[1:])
    )
    # No content lost: every word lands in at least one chunk.
    covered = " ".join(c.text for c in chunks)
    for i in range(200):
        assert f"word{i}" in covered


def test_empty_and_blank_text_yield_no_chunks():
    service = ChunkingService()
    assert service.split_text("") == []
    assert service.split_text("   \n\t  ") == []


def test_schema_rejects_empty_text_at_boundary():
    with pytest.raises(ValidationError):
        DocumentIn(text="")


def test_metadata_and_title_propagate_to_chunks():
    service = ChunkingService(chunk_size=50, chunk_overlap=10)
    doc = DocumentIn(
        doc_id="d3",
        title="FAQ",
        text="apple banana cherry date elderberry fig grape",
        metadata={"source": "seed"},
    )
    chunks = service.chunk_document(doc)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.title == "FAQ"
        assert chunk.metadata == {"source": "seed"}
        assert chunk.doc_id == "d3"


def test_missing_doc_id_is_generated():
    service = ChunkingService()
    chunks = service.chunk_document(DocumentIn(text="hello"))
    assert len(chunks) == 1
    assert chunks[0].doc_id
    assert chunks[0].chunk_id == f"{chunks[0].doc_id}:0"


def test_invalid_sizes_rejected():
    with pytest.raises(ValueError):
        ChunkingService(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        ChunkingService(chunk_size=100, chunk_overlap=150)
    with pytest.raises(ValueError):
        ChunkingService(chunk_size=0, chunk_overlap=0)
