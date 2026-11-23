"""Central app config. All settings come from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for configuration.

    Values load from environment (or a local .env file). Later chunks add
    chunking, retrieval, and re-ranking knobs here.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "raggy"
    environment: str = "dev"
    debug: bool = True

    # OpenAI (used from Chunk 3 onward; empty is fine for Chunk 1)
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Qdrant (used from Chunk 3 onward)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "raggy_docs"


def get_settings() -> Settings:
    """Factory so tests can override settings via dependency injection."""
    return Settings()
