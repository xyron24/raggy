"""Router tests: structured classification through the real parse path."""

from app.schemas import QueryKind, RouteDecision
from app.services.chat import ChatService
from app.services.router import ROUTER_SYSTEM_PROMPT, QueryRouter
from tests.fakes import FakeChatClient


def make_router(*decisions: RouteDecision) -> tuple[QueryRouter, FakeChatClient]:
    client = FakeChatClient(parsed_replies=list(decisions))
    chat = ChatService(api_key="test-key", client=client)
    return QueryRouter(chat), client


def test_route_returns_structured_decision():
    decision = RouteDecision(kind=QueryKind.VAGUE, reason="too short to answer directly")
    router, client = make_router(decision)
    assert router.route("refunds?") == decision
    (call,) = client.chat.completions.parse_calls
    assert call["response_format"] is RouteDecision
    assert call["messages"][0] == {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
    assert call["messages"][1] == {"role": "user", "content": "refunds?"}


def test_route_classifies_each_kind():
    decisions = [
        RouteDecision(kind=QueryKind.FACTUAL, reason="direct"),
        RouteDecision(kind=QueryKind.VAGUE, reason="underspecified"),
        RouteDecision(kind=QueryKind.COMPLEX, reason="multi-hop"),
    ]
    router, _ = make_router(*decisions)
    kinds = [router.route(f"question {i}").kind for i in range(3)]
    assert kinds == [QueryKind.FACTUAL, QueryKind.VAGUE, QueryKind.COMPLEX]


def test_needs_expansion_flags_non_factual():
    assert (
        QueryRouter.needs_expansion(RouteDecision(kind=QueryKind.FACTUAL, reason="x"))
        is False
    )
    assert (
        QueryRouter.needs_expansion(RouteDecision(kind=QueryKind.VAGUE, reason="x"))
        is True
    )
    assert (
        QueryRouter.needs_expansion(RouteDecision(kind=QueryKind.COMPLEX, reason="x"))
        is True
    )
