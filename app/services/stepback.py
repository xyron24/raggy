"""Step-Back Prompting for complex questions.

A specific multi-hop question ("what refund window applies to annual plans
bought during the sale?") embeds narrowly and misses background passages.
Step-Back asks the model for a broader question ("what are the refund
policies?") and retrieves for both, merging the results.
"""

from app.services.chat import ChatService

STEPBACK_SYSTEM_PROMPT = (
    "Given the user's specific question, write ONE broader, more general "
    "question that asks about the background knowledge needed to answer it. "
    "Write only the question, nothing else."
)


class StepBackService:
    """Generates broader step-back questions for expanded retrieval."""

    def __init__(self, chat: ChatService) -> None:
        self._chat = chat

    def step_back(self, query: str) -> str:
        """Return a broader question covering the query's background."""
        return self._chat.complete(STEPBACK_SYSTEM_PROMPT, query).strip()
