"""FastAPI entrypoint for the Raggy RAG backend."""

from fastapi import FastAPI

from app.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name, debug=settings.debug)


@app.get("/")
def read_root() -> dict:
    """Root endpoint. Confirms the API is up and names the service."""
    return {"service": settings.app_name, "environment": settings.environment}


@app.get("/health")
def health_check() -> dict:
    """Health check used by Docker, tests, and future load balancers."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.environment,
    }
