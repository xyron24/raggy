"""Document ingest and stats endpoints."""

from fastapi import APIRouter, Depends

from app.dependencies import get_ingest_service
from app.schemas import CollectionStats, DocumentIn, IngestResponse
from app.services.ingest import IngestService

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
    ingest: IngestService = Depends(get_ingest_service),
) -> CollectionStats:
    """Point count and shape info for the vector collection."""
    return ingest.stats()
