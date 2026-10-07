import json
from pathlib import Path

import pytest
from langchain.messages import HumanMessage

from app.core.llm import get_chat_model
from app.generation.answerer import AnswerOutcome
from app.knowledge.retriever import RetrievalResult
from langchain.tools import tool

from app.tools.ecommerce import query_logistics, query_order, query_product
from app.tools.ticket import build_create_ticket_tool

SAMPLES = Path(__file__).parent / "data" / "tool_routing_samples.jsonl"


def load_samples():
    return [json.loads(line) for line in SAMPLES.read_text().splitlines() if line.strip()]


pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not Path(".env").exists(), reason="需要 .env 真实上游"),
]


class _FakeRetriever:
    async def retrieve(self, q):
        return RetrievalResult(evidences=[], low_confidence=False)


class _FakeAnswerer:
    async def answer(self, q, evs):
        return AnswerOutcome(useful=True, answer="占位", citations=[])


@pytest.mark.parametrize("sample", load_samples(), ids=lambda s: s["text"][:12])
async def test_tool_routing_sample(session_factory, db_session, sample):
    """真实模型选型断言：五工具全量绑定，只看选了谁、不执行。"""
    from types import SimpleNamespace

    s = SimpleNamespace(retrieval_low_conf_threshold=0.45)
    tools = [
        query_order,
        query_product,
        query_logistics,
        build_query_faq_tool(db_session, _FakeRetriever(), _FakeAnswerer(), s, conversation_id=1),
        build_create_ticket_tool(db_session, conversation_id=1),  # 占位 cid，不执行
    ]
    model = get_chat_model().bind_tools(tools)
    ai = await model.ainvoke([HumanMessage(sample["text"])])
    names = [tc["name"] for tc in ai.tool_calls]
    expected = sample["expected_tool"]
    if expected is None:
        assert names == [], f"期望不调工具，实际: {names}"
    else:
        assert expected in names, f"期望 {expected}，实际: {names}"
