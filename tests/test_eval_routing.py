import json
from pathlib import Path

import pytest
from langchain.messages import HumanMessage

from app.core.config import get_settings
from app.core.llm import get_chat_model
from app.knowledge.embedder import build_embedder
from app.knowledge.milvus_store import MilvusStore
from app.tools.ecommerce import query_logistics, query_order, query_product
from app.tools.faq import build_query_faq_tool
from app.tools.ticket import build_create_ticket_tool

SAMPLES = Path(__file__).parent / "data" / "tool_routing_samples.jsonl"


def load_samples():
    return [json.loads(line) for line in SAMPLES.read_text().splitlines() if line.strip()]


pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not Path(".env").exists(), reason="需要 .env 真实上游"),
]


@pytest.mark.parametrize("sample", load_samples(), ids=lambda s: s["text"][:12])
async def test_tool_routing_sample(session_factory, db_session, sample):
    """真实模型选型断言：五工具全量绑定，只看选了谁、不执行。"""
    s = get_settings()
    embedder = build_embedder(s)
    store = MilvusStore(uri=s.milvus_uri, collection=s.milvus_collection)
    store.ensure_collection()
    tools = [
        query_order,
        query_product,
        query_logistics,
        build_query_faq_tool(embedder, store, s),
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
