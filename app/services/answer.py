"""Grounded question answering over retrieved passages."""

from app.schemas import CitedSource, QueryKind, QueryResponse, SearchResult
from app.services.chat import ChatService
from app.services.rerank import RerankerService
from app.services.retriever import RetrieverService
from app.services.router import QueryRouter

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered passages below. "
    "Every factual claim must cite the passage it came from with [n], "
    "where n is the passage number. If the passages do not contain the "
    "answer, say exactly: I don't know based on the provided context."
)

NO_PASSAGES_ANSWER = (
    "No relevant passages found. Ingest documents covering this topic first."
)


def format_context(results: list[SearchResult]) -> str:
    """Render passages as numbered [n] blocks the model must cite."""
    return "\n\n".join(
        f"[{i}]\n{hit.chunk.text}" for i, hit in enumerate(results, start=1)
    )


class AnswerService:
    """Routes, retrieves passages, and synthesizes a cited answer."""

    def __init__(
        self,
        retriever: RetrieverService,
        chat: ChatService,
        router: QueryRouter,
        reranker: RerankerService,
        hyde_enabled: bool = True,
        stepback_enabled: bool = True,
    ) -> None:
        self._retriever = retriever
        self._chat = chat
        self._router = router
        self._reranker = reranker
        self._hyde_enabled = hyde_enabled
        self._stepback_enabled = stepback_enabled

    def answer(self, query: str, top_k: int = 5) -> QueryResponse:
        """Answer a question with citations, or decline honestly.

        Routing runs before retrieval so each route can pick its own
        retrieval strategy: vague questions expand via HyDE, complex ones
        via Step-Back, factual ones use plain retrieval.
        """
        decision = self._router.route(query)
        use_hyde = self._hyde_enabled and decision.kind is QueryKind.VAGUE
        use_stepback = self._stepback_enabled and decision.kind is QueryKind.COMPLEX
        if use_stepback:
            results = self._retriever.retrieve_with_stepback(query, top_k=top_k)
        elif use_hyde:
            results = self._retriever.retrieve_with_hyde(query, top_k=top_k)
        else:
            results = self._retriever.retrieve(query, top_k=top_k)
        results = self._reranker.rerank(query, results)
        expanded = use_hyde or use_stepback
        if not results:
            return QueryResponse(
                answer=NO_PASSAGES_ANSWER,
                sources=[],
                route=decision.kind,
                confidence=0.0,
                expanded=expanded,
            )
        user_prompt = f"Passages:\n{format_context(results)}\n\nQuestion: {query}\nAnswer with citations:"
        text = self._chat.complete(SYSTEM_PROMPT, user_prompt).strip()
        return QueryResponse(
            answer=text or "I don't know based on the provided context.",
            sources=[
                CitedSource(
                    index=i,
                    chunk_id=hit.chunk.chunk_id,
                    doc_id=hit.chunk.doc_id,
                    title=hit.chunk.title,
                    text=hit.chunk.text,
                    score=hit.score,
                )
                for i, hit in enumerate(results, start=1)
            ],
            route=decision.kind,
            # Qdrant returns hits best-first, so the head is the max score.
            confidence=results[0].score,
            expanded=expanded,
        )
