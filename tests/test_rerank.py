"""Re-ranker tests: filtering, re-scoring, and verdict edge cases."""

import pytest

from app.schemas import Chunk, PassageScore, RerankVerdict, SearchResult
from app.services.rerank import RERANK_SYSTEM_PROMPT, RerankerService
from tests.fakes import FakeChatService


def make_candidates(n: int) -> list[SearchResult]:
    return [
        SearchResult(
            chunk=Chunk(
                chunk_id=f"d:{i}", doc_id="d", position=i, text=f"passage {i}"
            ),
            score=0.5,
        )
        for i in range(n)
    ]


def make_verdict(*pairs: tuple[int, float]) -> RerankVerdict:
    return RerankVerdict(
        scores=[PassageScore(index=i, score=s, reason="test") for i, s in pairs]
    )


def test_rerank_filters_and_resorts_by_fresh_scores():
    chat = FakeChatService(verdicts=[make_verdict((1, 0.9), (2, 0.1), (3, 0.7))])
    kept = RerankerService(chat, top_n=2, min_score=0.3).rerank("q?", make_candidates(3))
    assert [(h.chunk.chunk_id, h.score) for h in kept] == [("d:0", 0.9), ("d:2", 0.7)]


def test_rerank_cuts_to_top_n():
    chat = FakeChatService(verdicts=[make_verdict((1, 0.9), (2, 0.8), (3, 0.7))])
    kept = RerankerService(chat, top_n=2, min_score=0.0).rerank("q?", make_candidates(3))
    assert [h.chunk.chunk_id for h in kept] == ["d:0", "d:1"]


def test_rerank_empty_candidates_skips_llm():
    chat = FakeChatService()
    assert RerankerService(chat).rerank("q?", []) == []
    assert chat.structured_calls == []


def test_missing_index_scores_zero_and_drops():
    chat = FakeChatService(verdicts=[make_verdict((2, 0.8))])
    kept = RerankerService(chat, min_score=0.3).rerank("q?", make_candidates(2))
    assert [h.chunk.chunk_id for h in kept] == ["d:1"]


def test_out_of_range_index_ignored():
    chat = FakeChatService(verdicts=[make_verdict((1, 0.9), (99, 1.0))])
    kept = RerankerService(chat).rerank("q?", make_candidates(2))
    assert [(h.chunk.chunk_id, h.score) for h in kept] == [("d:0", 0.9)]


def test_rerank_prompt_wiring():
    chat = FakeChatService(verdicts=[make_verdict((1, 0.9))])
    RerankerService(chat).rerank("the question?", make_candidates(1))
    (system, user, model) = chat.structured_calls[0]
    assert system == RERANK_SYSTEM_PROMPT
    assert model is RerankVerdict
    assert "[1]\npassage 0" in user
    assert "the question?" in user


def test_invalid_knobs_rejected():
    chat = FakeChatService()
    with pytest.raises(ValueError):
        RerankerService(chat, top_n=0)
    with pytest.raises(ValueError):
        RerankerService(chat, min_score=1.5)
    with pytest.raises(ValueError):
        RerankerService(chat, min_score=-0.1)
