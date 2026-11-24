"""Document ingest and stats endpoints."""

from fastapi import APIRouter, Depends

from app.dependencies import get_ingest_service, get_vector_store
from app.schemas import CollectionStats, DocumentIn, IngestResponse
from app.services.ingest import IngestService
from app.services.vector_store import QdrantStore

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest", response_model=IngestResponse, status_code=201)
def ingest_document(
    document: DocumentIn,
    ingest: IngestService = Depends(get_ingest_service),
) -> IngestResponse:
    """Split, embed, and store a document. Re-ingesting an id overwrites."""
    return ingest.ingest(document)


@router.get("/stats", response_model=CollectionStats)
def collection_stats(
    store: QdrantStore = Depends(get_vector_store),
) -> CollectionStats:
    """Point count and shape info for the vector collection.

    Depends only on the store, so observability never needs an API key.
    """
    return CollectionStats(
        collection=store.collection,
        count=store.count(),
        vector_size=store.vector_size,
    )
