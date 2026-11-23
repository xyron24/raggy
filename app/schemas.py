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


class SearchResult(BaseModel):
    """One retrieval hit: the matched chunk plus its similarity score."""

    chunk: Chunk
    score: float = Field(description="Cosine similarity, higher is better.")


class IngestResponse(BaseModel):
    """Result of ingesting one document."""

    doc_id: str
    num_chunks: int = Field(ge=0)
    chunk_ids: list[str] = Field(description="Ids of stored chunks, in order.")


class CollectionStats(BaseModel):
    """Point count and shape info for the vector collection."""

    collection: str
    count: int = Field(ge=0)
    vector_size: int = Field(gt=0)


class QueryRequest(BaseModel):
    """A question against the ingested documents."""

    query: str = Field(min_length=1)
    top_k: int | None = Field(
        default=None, gt=0, le=20, description="Passages to use. Defaults to server setting."
    )


class CitedSource(BaseModel):
    """One passage cited by an answer, numbered to match [n] markers."""

    index: int = Field(ge=1, description="1-based citation number used in the answer.")
    chunk_id: str
    doc_id: str
    title: str | None = None
    text: str
    score: float


class QueryResponse(BaseModel):
    """Grounded answer plus the sources it cites."""

    answer: str
    sources: list[CitedSource] = Field(description="In citation order: sources[i].index == i + 1.")
