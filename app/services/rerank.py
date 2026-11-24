"""LLM re-ranking: score candidates precisely, keep only the best.

Vector search is tuned for recall (fast, approximate); the re-ranker is
tuned for precision (slower, exact). One structured call scores every
candidate, low scores are dropped, and survivors are re-sorted by their
fresh scores before synthesis.
"""

from app.schemas import RerankVerdict, SearchResult
from app.services.chat import ChatService

RERANK_SYSTEM_PROMPT = (
    "Score how relevant each numbered passage is to the user's question, "
    "from 0 (unrelated) to 1 (directly answers it). Score every passage, "
    "one score per passage number."
)


def format_candidates(candidates: list[SearchResult]) -> str:
    """Render candidates as numbered blocks for scoring."""
    return "\n\n".join(
        f"[{i}]\n{hit.chunk.text}" for i, hit in enumerate(candidates, start=1)
    )


class RerankerService:
    """Filters retrieval noise before answer synthesis."""

    def __init__(
        self, chat: ChatService, top_n: int = 3, min_score: float = 0.3
    ) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score must be within [0, 1]")
        self._chat = chat
        self.top_n = top_n
        self.min_score = min_score

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Return the best candidates by fresh relevance scores.

        Missing indices score 0 (dropped unless min_score is 0) and
        out-of-range indices are ignored, so a sloppy verdict degrades
        instead of crashing.
        """
        if not candidates:
            return []
        user_prompt = (
            f"Passages:\n{format_candidates(candidates)}\n\n"
            f"Question: {query}\nScore each passage:"
        )
        verdict = self._chat.complete_structured(
            RERANK_SYSTEM_PROMPT, user_prompt, RerankVerdict
        )
        scores: dict[int, float] = {}
        for item in verdict.scores:
            if 1 <= item.index <= len(candidates):
                scores[item.index] = max(scores.get(item.index, 0.0), item.score)
        ranked = [
            SearchResult(chunk=hit.chunk, score=scores.get(i, 0.0))
            for i, hit in enumerate(candidates, start=1)
        ]
        kept = [hit for hit in ranked if hit.score >= self.min_score]
        kept.sort(key=lambda hit: hit.score, reverse=True)
        return kept[: self.top_n]
