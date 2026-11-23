"""Intelligent query routing via structured model output."""

from app.schemas import QueryKind, RouteDecision
from app.services.chat import ChatService

ROUTER_SYSTEM_PROMPT = (
    "Classify the user question for a retrieval pipeline. Reply with exactly "
    "one kind: 'factual' for direct questions answerable from documents, "
    "'vague' for short or underspecified questions needing query expansion, "
    "'complex' for multi-hop or comparative questions needing broader "
    "retrieval. Also give a one-sentence reason."
)


class QueryRouter:
    """Classifies each question so retrieval can match its difficulty."""

    def __init__(self, chat: ChatService) -> None:
        self._chat = chat

    def route(self, query: str) -> RouteDecision:
        """Return the structured routing decision for a question."""
        return self._chat.complete_structured(
            ROUTER_SYSTEM_PROMPT, query, RouteDecision
        )

    @staticmethod
    def needs_expansion(decision: RouteDecision) -> bool:
        """True when plain retrieval is likely not enough."""
        return decision.kind in (QueryKind.VAGUE, QueryKind.COMPLEX)
