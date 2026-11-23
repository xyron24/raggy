"""Pydantic models for documents and their text chunks."""

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    """A document submitted for ingestion.

    `doc_id` is optional: callers that track their own ids pass one for
    idempotent re-ingests, and everyone else gets a generated id.
    """

    doc_id: str | None = Field(
        default=None, description="Caller-provided stable id. Generated when omitted."
    )
    title: str | None = Field(
        default=None, description="Human-readable title, propagated to every chunk."
    )
    text: str = Field(min_length=1, description="Full document text to split.")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Custom key-value metadata, propagated to every chunk.",
    )


class Chunk(BaseModel):
    """One passage produced by splitting a document."""

    chunk_id: str = Field(description="Deterministic id: '<doc_id>:<position>'.")
    doc_id: str
    position: int = Field(
        ge=0, description="Zero-based order of the chunk inside its document."
    )
    text: str
    title: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
