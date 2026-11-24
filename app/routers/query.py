"""Grounded question-answering endpoint."""

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_answer_service
from app.schemas import QueryRequest, QueryResponse
from app.services.answer import AnswerService

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def ask_question(
    request: QueryRequest,
    answer: AnswerService = Depends(get_answer_service),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """Answer a question from ingested documents, with cited sources."""
    top_k = request.top_k if request.top_k is not None else settings.top_k_retrieve
    return answer.answer(request.query, top_k=top_k)
