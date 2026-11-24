"""FastAPI dependency wiring for services.

Each function builds one service from settings. Tests replace any of them
via `app.dependency_overrides`, so no test needs real keys or servers.
"""

from fastapi import Depends, HTTPException

from app.config import Settings, get_settings
from app.services.answer import AnswerService
from app.services.chat import ChatService
from app.services.chunking import ChunkingService
from app.services.embeddings import EmbeddingService
from app.services.hyde import HyDEService
from app.services.ingest import IngestService
from app.services.retriever import RetrieverService
from app.services.router import QueryRouter
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


def get_chat_service(
    settings: Settings = Depends(get_settings),
) -> ChatService:
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503, detail="chat not configured: set OPENAI_API_KEY"
        )
    return ChatService(
        api_key=settings.openai_api_key, model=settings.openai_chat_model
    )


def get_hyde_service(
    chat: ChatService = Depends(get_chat_service),
) -> HyDEService:
    return HyDEService(chat=chat)


def get_retriever_service(
    embeddings: EmbeddingService = Depends(get_embedding_service),
    store: QdrantStore = Depends(get_vector_store),
    hyde: HyDEService = Depends(get_hyde_service),
) -> RetrieverService:
    return RetrieverService(embeddings=embeddings, store=store, hyde=hyde)


def get_router_service(
    chat: ChatService = Depends(get_chat_service),
) -> QueryRouter:
    return QueryRouter(chat=chat)


def get_answer_service(
    retriever: RetrieverService = Depends(get_retriever_service),
    chat: ChatService = Depends(get_chat_service),
    router: QueryRouter = Depends(get_router_service),
    settings: Settings = Depends(get_settings),
) -> AnswerService:
    return AnswerService(
        retriever=retriever,
        chat=chat,
        router=router,
        hyde_enabled=settings.enable_hyde,
    )
