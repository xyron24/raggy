"""FastAPI dependency wiring for services.

Each function builds one service from settings. Tests replace any of them
via `app.dependency_overrides`, so no test needs real keys or servers.
"""

from fastapi import Depends, HTTPException

from app.config import Settings, get_settings
from app.services.chunking import ChunkingService
from app.services.embeddings import EmbeddingService
from app.services.ingest import IngestService
from app.services.vector_store import QdrantStore


def get_chunking_service(
    settings: Settings = Depends(get_settings),
) -> ChunkingService:
    return ChunkingService(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )


def get_embedding_service(
    settings: Settings = Depends(get_settings),
) -> EmbeddingService:
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503, detail="embeddings not configured: set OPENAI_API_KEY"
        )
    return EmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.openai_embed_model,
        batch_size=settings.embed_batch_size,
    )


def get_vector_store(
    settings: Settings = Depends(get_settings),
) -> QdrantStore:
    return QdrantStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        vector_size=settings.embedding_dim,
    )


def get_ingest_service(
    chunking: ChunkingService = Depends(get_chunking_service),
    embeddings: EmbeddingService = Depends(get_embedding_service),
    store: QdrantStore = Depends(get_vector_store),
) -> IngestService:
    return IngestService(chunking=chunking, embeddings=embeddings, store=store)
