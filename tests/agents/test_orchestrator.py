from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.agents.orchestrator import Orchestrator
from app.db.models import Message


@dataclass
class FakeChunk:
    text: str = ""
    tool_call_chunks: list = field(default_factory=list)


class FakeModel:
    """按调用次序脚本化产出：bind_tools 后第一次 astream 用 script[0]，第二次用 script[1]。"""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.calls.append(list(messages))  # 记录消息对象，测试按需取 .type/.content
        step = self.script[len(self.calls) - 1]
        for piece in step:
            yield piece


class FakeRegistry:
    def __init__(self):
        self.executed = []

    def all(self):
        return []

    def has(self, name):
        return True  # 与真实 ToolRegistry.has 同接口

    async def execute(self, name, args):
        self.executed.append((name, args))
        return {"ok": True, "data": {"fake": "result"}}


def make_orchestrator(model, session_factory, registry=None):
    return Orchestrator(
        model=model,
        registry_factory=lambda session, cid: registry or FakeRegistry(),
        session_factory=session_factory,
        settings=SimpleNamespace(token_budget=4000),
    )


async def test_direct_answer_no_tool(session_factory, db_session):
    model = FakeModel([[FakeChunk(text="你好"), FakeChunk(text="呀")]])
    o = make_orchestrator(model, session_factory)
    events = [e async for e in o.run(user_id="u1", conversation_id=None, message="你好")]
    kinds = [k for k, _ in events]
    assert kinds[0] == "meta" and kinds[-1] == "done"
    assert "tool" not in kinds  # 没有工具帧
    assert "".join(v.get("text", "") for k, v in events if k == "delta") == "你好呀"
    assert len(model.calls) == 1  # 只调了一次模型
    # 落库：user + assistant
    rows = (await db_session.execute(select(Message).order_by(Message.id))).scalars().all()
    assert [r.role for r in rows] == ["user", "assistant"]
    assert rows[1].content == "你好呀"


@pytest.mark.usefixtures("db_session")
async def test_tool_path_feed_back_and_stream(session_factory, db_session):
    tc = {"name": "query_faq", "args": '{"keyword": "退货"}', "id": "call_1", "index": 0, "type": "tool_call_chunk"}
    model = FakeModel(
        [
            [FakeChunk(tool_call_chunks=[tc])],  # 第一轮：要工具
            [FakeChunk(text="根据"), FakeChunk(text="政策…")],  # 第二轮：流式收敛
        ]
    )
    reg = FakeRegistry()
    o = make_orchestrator(model, session_factory, registry=reg)
    events = [
        e
        async for e in o.run(
            user_id="u1", conversation_id=None, message="退货政策是什么"
        )
    ]

    kinds = [k for k, _ in events]
    assert kinds[0] == "meta" and kinds[-1] == "done"
    tool_frames = [v for k, v in events if k == "tool"]
    assert len(tool_frames) == 2  # running + done
    assert tool_frames[0]["status"] == "running" and tool_frames[0]["tool"] == "query_faq"
    assert tool_frames[1]["status"] == "done"
    assert reg.executed == [("query_faq", {"keyword": "退货"})]
    # 第二次调用收到回灌：system → …human → ai(申请, type=ai) → tool(结果)
    second = model.calls[1]
    assert second[0].type == "system"  # 客服角色每轮由编排器拼接，不落库
    assert second[-1].type == "tool"
    assert second[-2].type == "ai"
    assert len(model.calls) == 2
    deltas = "".join(v.get("text", "") for k, v in events if k == "delta")
    assert deltas == "根据政策…"  # delta 只来自第二轮
    # 落库：user / assistant(tool_calls) / tool / assistant 四行
    rows = (await db_session.execute(select(Message).order_by(Message.id))).scalars().all()
    assert [r.role for r in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[1].tool_calls[0]["name"] == "query_faq"
    assert rows[2].tool_call_id == "call_1"
    assert '"ok"' in rows[2].content  # 回灌内容是注册表包装后的 JSON


@pytest.mark.usefixtures("db_session")
async def test_tool_error_still_converges(session_factory, db_session):
    class BoomRegistry(FakeRegistry):
        async def execute(self, name, args):
            return {"ok": False, "error": "查询超时"}

    tc = {"name": "query_order", "args": '{"order_no": "1001"}', "id": "call_9", "index": 0, "type": "tool_call_chunk"}
    model = FakeModel(
        [
            [FakeChunk(tool_call_chunks=[tc])],
            [FakeChunk(text="查询失败了，抱歉")],
        ]
    )
    o = make_orchestrator(model, session_factory, registry=BoomRegistry())
    events = [e async for e in o.run(user_id="u1", conversation_id=None, message="订单1001呢")]
    kinds = [k for k, _ in events]
    assert kinds[-1] == "done"  # 工具失败不炸会话，仍收敛
    tool_frames = [v for k, v in events if k == "tool"]
    assert tool_frames[1]["ok"] is False
    # 失败结果也回灌给模型
    second = model.calls[1]
    assert second[-1].type == "tool"


def test_aggregate_parallel_tool_calls():
    from app.agents.orchestrator import _aggregate_tool_calls

    chunks = [
        [{"name": "query_order", "args": '{"order', "id": "a1", "index": 0}],
        [{"name": "query_faq", "args": '{"keyword"', "id": "a2", "index": 1}],
        [{"args": '_no": "1001"}', "index": 0}],
        [{"args": ': "退货"}', "index": 1}],
    ]
    calls = _aggregate_tool_calls(chunks)
    assert calls == [
        {"name": "query_order", "args": {"order_no": "1001"}, "id": "a1"},
        {"name": "query_faq", "args": {"keyword": "退货"}, "id": "a2"},
    ]


async def test_leading_text_persisted_in_tool_mode(session_factory, db_session):
    """评审 Minor 4：混发场景下，工具调用前已播报的正文必须落库。"""
    model = FakeModel(
        [
            [
                FakeChunk(text="让我查一下"),
                FakeChunk(
                    tool_call_chunks=[
                        {"name": "query_faq", "args": '{"keyword": "退货"}', "id": "c1", "index": 0}
                    ]
                ),
            ],
            [FakeChunk(text="根据政策…")],
        ]
    )
    o = make_orchestrator(model, session_factory)
    _events = [e async for e in o.run(user_id="u5", conversation_id=None, message="退货政策")]

    rows = (await db_session.execute(select(Message).order_by(Message.id))).scalars().all()
    assert rows[1].content == "让我查一下"  # 先行正文落库，下轮模型可见
    assert rows[1].tool_calls[0]["name"] == "query_faq"
