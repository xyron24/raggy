"""FastAPI entrypoint for the Raggy RAG backend."""

from fastapi import FastAPI

from app.config import get_settings
from app.routers.documents import router as documents_router
from app.routers.query import router as query_router

settings = get_settings()

app = FastAPI(title=settings.app_name, debug=settings.debug)
app.include_router(documents_router)
app.include_router(query_router)


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
