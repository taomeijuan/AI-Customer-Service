"""ch07 T11 场景①②④：演示配置下 24 轮级联（降级→摘要→靠梗概答对）+ 日志可观测。

真实 LLM：resolver/intent/摘要/末轮 Agent；常规轮 Agent 与检索用桩压成本与时延。
eval 标记：需 .env 与 MySQL 测试库。
"""

import logging

import pytest
from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from app.core.config import Settings
from app.core.llm import get_chat_model
from app.db.models import Conversation, ConversationSummary
from app.extraction.service import build_structured_model
from app.knowledge.retriever import RetrievalResult
from app.main import create_app
from app.memory.budget import compute_budget
from app.memory.summarizer import Summarizer
from app.repositories.conversations import ConversationsRepo
from app.repositories.messages import MessagesRepo
from app.workflow.agent_node import build_agent_node
from app.workflow.graph import build_workflow
from app.workflow.intent import LangChainIntentClassifier
from app.workflow.resolver import ResolutionSchema, Resolver

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not __import__("pathlib").Path(".env").exists(), reason="需要 .env 真实上游"),
]

DEMO = dict(
    model_context_window=18000, max_output_tokens=2000, max_user_input_tokens=2000,
    tool_result_max_tokens=1200, max_agent_steps=3, rerank_top_k=5,
)

class _SwitchAgent:
    """前段桩答（可控长文本），末轮切真 Agent。"""

    def __init__(self, real_builder):
        self.calls = []
        self.real = None
        self._real_builder = real_builder

    async def __call__(self, state):
        self.calls.append(dict(state))
        if len(self.calls) >= 24:
            if self.real is None:
                self.real = self._real_builder()
            return await self.real(state)
        return {
            "final_text": f"关于订单1001扫地机器人退款诉求的答复轮{len(self.calls)}：" + "详细说明内容" * 40,
            "messages": [HumanMessage(state["query"]), AIMessage(content="占位")],
            "evidence": state.get("evidence", []),
        }


class _StubRetriever:
    async def retrieve(self, query, strategy="hybrid_rerank", category_prefix=None):
        return RetrievalResult(evidences=[], low_confidence=False)

    async def retrieve_multi(self, queries, strategy="hybrid_rerank", category_prefix=None):
        return RetrievalResult(evidences=[], low_confidence=False)


@pytest.mark.usefixtures("db_session")
async def test_demo_config_cascade_and_summary_answer(session_factory, db_session, caplog):
    settings = Settings(_env_file=".env", **DEMO)
    budget = compute_budget(settings)
    assert (budget.history, budget.layer1, budget.layer2) == (5650, 3954, 1696)

    model = get_chat_model()
    app = create_app()
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.context_budget = budget

    scheduled = []

    agent = _SwitchAgent(
        lambda: build_agent_node(model, [], settings)  # 末轮：无工具纯靠 material 答
    )
    summarizer = Summarizer(model, session_factory, settings)
    _orig_schedule = summarizer.schedule

    def _schedule(cid, batch, projection):
        t = _orig_schedule(cid, batch, projection)
        if t is not None:
            scheduled.append(t)
        return t

    summarizer.schedule = _schedule
    app.state.summarizer = summarizer

    app.state.workflow = build_workflow(
        retriever=_StubRetriever(),
        agent_node=agent,
        intent_classifier=LangChainIntentClassifier(model),
        session_factory=session_factory,
        resolver=Resolver(build_structured_model(model, ResolutionSchema)),
        checkpointer=InMemorySaver(),
    )

    import httpx

    transport = httpx.ASGITransport(app=app)
    cid = None
    with caplog.at_level(logging.INFO):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            for k in range(1, 24):
                r = await c.post(
                    "/api/chat/stream",
                    json={
                        "user_id": "ctx-eval",
                        "message": f"追问{k}：我买的订单1001扫地机器人一直没发货想退款，" + "补充背景信息" * 12,
                        "conversation_id": cid,
                    },
                )
                if k == 1:
                    cid = [v for e, v in _events(r.text) if e == "meta"][0]["conversation_id"]
                # 轮间收割摘要任务：确保投影在后续轮可读（摘要不阻塞当轮的证明）
                if scheduled:
                    await asyncio_gather(scheduled)
                    scheduled.clear()
            # 末轮：不带任何订单号，靠梗概答对
            rf = await c.post(
                "/api/chat/stream",
                json={"user_id": "ctx-eval", "message": "最开始那个订单后来怎么说？", "conversation_id": cid},
            )
    await asyncio_gather(scheduled)

    log = caplog.text
    assert "layer1 demote" in log, "层1超预算应有降级留痕"
    assert "summary trigger" in log, "层2超预算应有摘要触发"
    assert "summary done" in log, "摘要任务应有完成留痕"
    assert "history_ctx" in log and "model_ctx" in log, "双观测日志都要在"

    async with session_factory() as session:
        segs = (
            await session.execute(
                select(ConversationSummary).where(ConversationSummary.conversation_id == cid)
            )
        ).scalars().all()
        assert len(segs) >= 1, "至少压出一段梗概"
        conv = await session.get(Conversation, cid)
        assert conv.summary and "1001" in conv.summary, "梗概里必须存着订单号"
        assert conv.summary_upto_msg_id and conv.summary_upto_msg_id > 0

    finals = [v for e, v in _events(rf.text) if e == "delta"]
    answer = "".join(v["text"] for v in finals)
    assert "1001" in answer, f"末轮回答必须靠梗概带出订单号：{answer[:120]}"


def _events(text):
    import json

    out = []
    for block in text.split("\n\n"):
        ev = da = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                da = line[5:].strip()
        if da:
            try:
                out.append((ev, json.loads(da)))
            except Exception:
                pass
    return out


async def asyncio_gather(tasks):
    import asyncio

    if tasks:
        await asyncio.wait(tasks)
