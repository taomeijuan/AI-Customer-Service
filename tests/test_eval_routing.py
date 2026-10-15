import json
from pathlib import Path
from typing import Annotated, Literal

import pytest
from langchain.messages import HumanMessage
from langchain.tools import tool
from pydantic import Field

from app.core.llm import get_chat_model
from app.generation.answerer import AnswerOutcome
from app.knowledge.retriever import RetrievalResult
from app.tools.ecommerce import query_logistics, query_order, query_product
from app.tools.plugins.ticket import _DESCRIPTION as CREATE_TICKET_DESCRIPTION

SAMPLES = Path(__file__).parent / "data" / "tool_routing_samples.jsonl"


@tool(description=CREATE_TICKET_DESCRIPTION)
def create_ticket(
    description: Annotated[
        str,
        Field(description="问题描述，必须是用户原话。用户没说具体问题就不要调用本工具。"),
    ],
    ticket_type: Annotated[
        Literal["售后", "投诉", "咨询"],
        Field(description="只能填用户明确说过的售后、投诉或咨询。没说类型就不要调用，不要猜测。"),
    ],
) -> str:
    """eval 只看选型，不执行。"""
    raise RuntimeError("eval 不执行工具")


@tool
def query_warranty(order_no: str) -> str:
    """查询订单商品的质保状态与到期日。当用户询问保修、质保、过保时调用。"""
    raise RuntimeError("eval 不执行工具")


@tool
def query_return(order_no: str) -> str:
    """查询订单退货进度。当用户询问退货、退款到哪了时调用。"""
    raise RuntimeError("eval 不执行工具")


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
async def test_tool_routing_sample(sample):
    """真实模型选型断言：查询工具与建单桩全量绑定，只看选了谁、不执行。"""
    tools = [
        query_order,
        query_product,
        query_logistics,
        create_ticket,
        query_warranty,
        query_return,
    ]
    model = get_chat_model().bind_tools(tools)
    ai = await model.ainvoke([HumanMessage(sample["text"])])
    names = [tc["name"] for tc in ai.tool_calls]
    expected = sample["expected_tool"]
    if expected is None:
        assert names == [], f"期望不调工具，实际: {names}"
    else:
        assert expected in names, f"期望 {expected}，实际: {names}"
