"""Central app config. All settings come from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for configuration.

    Values load from environment (or a local .env file). Later phases add
    chunking, retrieval, and re-ranking knobs here.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "raggy"
    environment: str = "dev"
    debug: bool = True

    # OpenAI (wired up in Phase 3; empty is fine until then)
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Qdrant (wired up in Phase 3)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "raggy_docs"

    # Chunking (used from Phase 2 onward)
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Embeddings (used from Phase 3 onward; dim must match the model)
    embed_batch_size: int = 100
    embedding_dim: int = 1536

    # Retrieval (used from Phase 5 onward)
    default_top_k: int = 5


def get_settings() -> Settings:
    """Factory so tests can override settings via dependency injection."""
    return Settings()
