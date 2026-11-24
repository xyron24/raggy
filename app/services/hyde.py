"""HyDE: Hypothetical Document Embeddings for vague queries.

A short question ("refunds?") embeds poorly because questions and documents
use different language. HyDE asks the model to imagine a plausible answer
passage first; that passage looks like document language, so its embedding
retrieves better.
"""

from app.services.chat import ChatService

HYDE_SYSTEM_PROMPT = (
    "Write a short passage that directly answers the user's question, as if "
    "it were an excerpt from a document in the knowledge base. Write only "
    "the passage, no preamble and no citations."
)


class HyDEService:
    """Generates hypothetical answer passages for HyDE retrieval."""

    def __init__(self, chat: ChatService) -> None:
        self._chat = chat

    def imagine(self, query: str) -> str:
        """Return a hypothetical answer passage for the query."""
        return self._chat.complete(HYDE_SYSTEM_PROMPT, query).strip()
