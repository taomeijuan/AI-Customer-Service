"""T8：子图 interrupt 探针，以及建工单确认/取消的父图桥接。"""

import datetime
import re

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from sqlalchemy import func, select

from app.core.config import Settings
from app.db.models import Ticket, ToolAuditLog
from app.knowledge.retriever import RetrievalResult
from app.repositories.conversations import ConversationsRepo
from app.tools.registry import ToolRegistry, scan_plugins
from app.workflow.agent_node import build_agent_node, paused_agents
from app.workflow.graph import build_workflow
from app.workflow.intent import ClassifyOutcome

_PAYLOAD = {"type": "ticket_preview", "ticket_type": "售后", "description": "耳机没声音"}


@tool
async def preview_ticket() -> str:
    """暂停并交出工单预览。探针工具，不写库。"""
    decision = interrupt(_PAYLOAD)
    return str(decision)


class _ScriptedModel(BaseChatModel):
    """先发一次工具调用，再吐最终文本。"""

    steps: list
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "ticket-interrupt-probe"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        step = self.steps[min(self.calls - 1, len(self.steps) - 1)]
        if isinstance(step, tuple):
            name, args = step
            msg = AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": "c1", "type": "tool_call"}],
            )
        else:
            msg = AIMessage(content=step)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _interrupt_value(result):
    """兼容 dict['__interrupt__'] 与 GraphOutput.interrupts。"""
    bag = None
    if isinstance(result, dict):
        bag = result.get("__interrupt__")
        if bag:
            return bag[0].value, 'result["__interrupt__"][0].value'
    interrupts = getattr(result, "interrupts", None)
    if interrupts:
        return interrupts[0].value, "result.interrupts[0].value"
    return None, None


async def test_react_tool_interrupt_surfaces():
    """假工具 interrupt 后，ainvoke 结果必须带上 ticket_preview 载荷。"""
    agent = create_react_agent(
        _ScriptedModel(steps=[("preview_ticket", {}), "已提交预览"]),
        [preview_ticket],
        checkpointer=InMemorySaver(),
        version="v2",
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage("耳机没声音")]},
        config={"configurable": {"thread_id": "probe-ticket"}},
    )
    value, shape = _interrupt_value(result)
    print(f"PROBE_INTERRUPT_SHAPE {shape} result_type={type(result).__name__}")
    assert value == _PAYLOAD, (
        f"interrupt shape={shape!r} result_type={type(result).__name__} result={result!r}"
    )


# --- 图测试：真引擎 + 真库。探针已确认载荷在 result["__interrupt__"][0].value ---


@pytest.fixture(autouse=True)
def _reset_paused_agents():
    paused_agents.clear()
    yield
    paused_agents.clear()


class _TicketModel(BaseChatModel):
    """按已出现的 ToolMessage 数取下一步：0 → 调工具，之后 → 最终文本。"""

    steps: list
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "ticket-graph-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))
        step = self.steps[min(tool_count, len(self.steps) - 1)]
        if isinstance(step, tuple):
            name, args = step
            msg = AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": f"c{tool_count + 1}", "type": "tool_call"}],
            )
        else:
            msg = AIMessage(content=step)
        return ChatResult(generations=[ChatGeneration(message=msg)])


class _OrderClassifier:
    async def classify_detail(self, query):
        return ClassifyOutcome(intent="订单", confidence=0.9)

    async def classify(self, query):
        return "订单"


class _QuietRetriever:
    async def retrieve(self, query, strategy="hybrid_rerank", category_prefix=None):
        return RetrievalResult(evidences=[], low_confidence=False)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        max_agent_steps=8,
        tool_timeout=3.0,
        tool_retries=1,
        tool_delay_tools="",
        tool_delay_seconds=0,
    )


def _parent(session_factory, llm):
    registry = ToolRegistry()
    registry.session_factory = session_factory
    scan_plugins(registry)
    node = build_agent_node(
        llm,
        [],
        _settings(),
        registry=registry,
        session_factory=session_factory,
    )
    return build_workflow(
        retriever=_QuietRetriever(),
        agent_node=node,
        intent_classifier=_OrderClassifier(),
        session_factory=session_factory,
        checkpointer=InMemorySaver(),
    )


async def _cid(db_session, user_id: str) -> int:
    return await ConversationsRepo(db_session).ensure_conversation(user_id=user_id)


async def _ticket_count(session_factory) -> int:
    async with session_factory() as session:
        count = await session.scalar(select(func.count()).select_from(Ticket))
        return int(count or 0)


async def _audits(session_factory, tool_name: str = "create_ticket") -> list[ToolAuditLog]:
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(ToolAuditLog)
                .where(ToolAuditLog.tool_name == tool_name)
                .order_by(ToolAuditLog.id)
            )
        ).scalars().all()
        return list(rows)


def _preview(result) -> dict:
    value, shape = _interrupt_value(result)
    assert value is not None, f"parent interrupt missing shape={shape!r} result={result!r}"
    return value


def _mentions(out: dict, ticket_no: str) -> bool:
    if ticket_no in (out.get("final_text") or ""):
        return True
    for ev in out.get("evidence") or []:
        if ticket_no in (ev.get("answer") or ""):
            return True
    for message in out.get("messages") or []:
        content = getattr(message, "content", "")
        if isinstance(content, str) and ticket_no in content:
            return True
    return False


@pytest.mark.usefixtures("db_session")
async def test_valid_create_ticket_pauses_without_write(db_session, session_factory):
    """合法参数只停在 ticket_preview：不落库，暂停本身不记成功。"""
    cid = await _cid(db_session, "t8-pause")
    before = await _ticket_count(session_factory)
    llm = _TicketModel(
        steps=[
            ("create_ticket", {"description": "耳机没声音", "ticket_type": "售后"}),
            "已创建",
        ]
    )
    graph = _parent(session_factory, llm)
    result = await graph.ainvoke(
        {"query": "耳机没声音，帮我建个售后工单", "conversation_id": cid, "messages": []},
        config={"configurable": {"thread_id": f"t8-pause-{cid}"}},
    )
    payload = _preview(result)
    assert payload["type"] == "ticket_preview"
    assert payload["ticket_type"] == "售后"
    assert payload["description"] == "耳机没声音"
    assert await _ticket_count(session_factory) == before
    assert all(row.status != "成功" for row in await _audits(session_factory))


@pytest.mark.usefixtures("db_session")
async def test_confirm_resume_inserts_ticket(db_session, session_factory):
    """同一父图线程 resume confirm 后落库，工单号出现在工具结果或最终回答里。"""
    cid = await _cid(db_session, "t8-confirm")
    llm = _TicketModel(
        steps=[
            ("create_ticket", {"description": "耳机没声音", "ticket_type": "售后"}),
            "工单已创建",
        ]
    )
    graph = _parent(session_factory, llm)
    config = {"configurable": {"thread_id": f"t8-confirm-{cid}"}}
    paused = await graph.ainvoke(
        {"query": "耳机没声音，帮我建个售后工单", "conversation_id": cid, "messages": []},
        config=config,
    )
    assert _preview(paused)["type"] == "ticket_preview"

    done = await graph.ainvoke(Command(resume={"action": "confirm"}), config=config)
    today = datetime.date.today().strftime("%Y%m%d")
    async with session_factory() as session:
        rows = (await session.execute(select(Ticket))).scalars().all()
    assert len(rows) == 1
    ticket_no = rows[0].ticket_no
    assert re.fullmatch(rf"T{today}\d+", ticket_no)
    assert rows[0].conversation_id == cid
    assert _mentions(done, ticket_no)


@pytest.mark.usefixtures("db_session")
async def test_cancel_resume_denies_without_insert(db_session, session_factory):
    """另一会话取消：不落库，审计为权限拒绝，回灌「用户取消了建单」。"""
    cid = await _cid(db_session, "t8-cancel")
    before = await _ticket_count(session_factory)
    llm = _TicketModel(
        steps=[
            ("create_ticket", {"description": "耳机没声音", "ticket_type": "售后"}),
            "已取消",
        ]
    )
    graph = _parent(session_factory, llm)
    config = {"configurable": {"thread_id": f"t8-cancel-{cid}"}}
    await graph.ainvoke(
        {"query": "耳机没声音，帮我建个售后工单", "conversation_id": cid, "messages": []},
        config=config,
    )
    await graph.ainvoke(Command(resume={"action": "cancel"}), config=config)

    assert await _ticket_count(session_factory) == before
    rows = await _audits(session_factory)
    assert len(rows) == 1
    assert rows[0].status == "权限拒绝"
    assert rows[0].retry_count == 0
    assert rows[0].error_message is not None and "用户取消了建单" in rows[0].error_message


@pytest.mark.usefixtures("db_session")
async def test_empty_description_does_not_interrupt(db_session, session_factory):
    """空描述被校验拦下，不弹出 ticket_preview。"""
    cid = await _cid(db_session, "t8-invalid")
    before = await _ticket_count(session_factory)
    llm = _TicketModel(
        steps=[
            ("create_ticket", {"description": "", "ticket_type": "售后"}),
            "请补充问题描述",
        ]
    )
    graph = _parent(session_factory, llm)
    result = await graph.ainvoke(
        {"query": "帮我建个工单", "conversation_id": cid, "messages": []},
        config={"configurable": {"thread_id": f"t8-invalid-{cid}"}},
    )
    assert not (result.get("__interrupt__") if isinstance(result, dict) else getattr(result, "interrupts", None))
    assert await _ticket_count(session_factory) == before
    rows = await _audits(session_factory)
    assert [row.status for row in rows] == ["校验拦下"]


@pytest.mark.usefixtures("db_session")
async def test_missing_required_and_extra_fields_caught_by_engine_validation(db_session, session_factory):
    """M1 回归：缺必填字段或多余字段不能被 Pydantic 提前拦截/静默剥离，必须进 validate_args 记『校验拦下』。"""
    cid = await _cid(db_session, "t8-m1-schema")
    before = await _ticket_count(session_factory)
    llm = _TicketModel(
        steps=[
            ("create_ticket", {"ticket_type": "售后"}),
            ("create_ticket", {"description": "坏了", "ticket_type": "售后", "extra": 1}),
            "请补充有效信息",
        ]
    )
    graph = _parent(session_factory, llm)
    result = await graph.ainvoke(
        {"query": "帮我建个工单", "conversation_id": cid, "messages": []},
        config={"configurable": {"thread_id": f"t8-m1-{cid}"}},
    )
    assert not (result.get("__interrupt__") if isinstance(result, dict) else getattr(result, "interrupts", None))
    assert await _ticket_count(session_factory) == before
    rows = await _audits(session_factory)
    assert [row.status for row in rows] == ["校验拦下", "校验拦下"]
    assert "缺少必填字段 description" in (rows[0].error_message or "")
    assert "不允许多余字段 extra" in (rows[1].error_message or "")
    tool_msgs = [m for m in result.get("messages", []) if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2
    assert "参数不合法：缺少必填字段 description" in tool_msgs[0].content
    assert "参数不合法：不允许多余字段 extra" in tool_msgs[1].content


@pytest.mark.usefixtures("db_session")
async def test_tool_frame_handler_args_ok_and_interrupt_suppression(db_session, session_factory):
    """M3 回归：
    1) on_tool_start 从 kwargs['inputs'] 提取真实 args（非空 dict）；
    2) on_tool_end 根据引擎返回 ok: false 推 ok=False；
    3) create_ticket 触发 GraphInterrupt 时不推 bogus done(ok=False) 帧。
    """
    from app.workflow.agent_node import _ToolFrameHandler
    from langgraph.errors import GraphInterrupt

    emitted: list[dict] = []
    handler = _ToolFrameHandler(lambda frame: emitted.append(frame))

    # 1) kwargs["inputs"] 含 dict，input_str 是 Python 单引号 repr
    await handler.on_tool_start(
        {"name": "query_order"},
        "{'order_no': '1001', 'tool_call_id': 'c1'}",
        inputs={"order_no": "1001", "tool_call_id": "c1"},
    )
    assert emitted[-1] == {
        "tool": {"tool": "query_order", "args": {"order_no": "1001"}, "status": "running", "ok": True}
    }

    # 2) 引擎返回 {"ok": false, ...} 时 done 帧 ok 应为 False
    await handler.on_tool_end(
        ToolMessage(content='{"ok": false, "error_type": "not_found", "error": "没有查到对应记录"}', tool_call_id="c1")
    )
    assert emitted[-1] == {
        "tool": {"tool": "query_order", "args": {"order_no": "1001"}, "status": "done", "ok": False}
    }

    # 3) GraphInterrupt 触发 on_tool_error 时静默弹栈，不发失败 done 帧
    await handler.on_tool_start(
        {"name": "create_ticket"},
        "{'description': '耳机坏了', 'ticket_type': '售后'}",
        inputs={"description": "耳机坏了", "ticket_type": "售后", "tool_call_id": "c2"},
    )
    count_before = len(emitted)
    await handler.on_tool_error(GraphInterrupt())
    assert len(emitted) == count_before
    assert handler._stack == []

    # 4) MCP 适配器工具在 _EngineTool 内部嵌套 ainvoke 时，不重复发 running/done 帧
    await handler.on_tool_start(
        {"name": "query_logistics"},
        "{'order_no': '1001'}",
        inputs={"order_no": "1001", "tool_call_id": "c3"},
    )
    assert len(emitted) == count_before + 1
    await handler.on_tool_start(
        {"name": "query_logistics"},
        "{'order_no': '1001'}",
        inputs={"order_no": "1001"},
    )
    assert len(emitted) == count_before + 1  # 内层工具 start 不重发
    await handler.on_tool_end([{"type": "text", "text": '{"order_no": "1001"}'}])
    assert len(emitted) == count_before + 1  # 内层工具 end 不重发
    await handler.on_tool_end(ToolMessage(content='{"ok": true, "data": {"order_no": "1001"}}', tool_call_id="c3"))
    assert len(emitted) == count_before + 2
    assert emitted[-1]["tool"]["status"] == "done" and emitted[-1]["tool"]["ok"] is True


@pytest.mark.usefixtures("db_session")
async def test_paused_agents_popped_immediately_on_resume_even_if_cancelled(db_session, session_factory):
    """M4 回归：resume 唤醒后立即从 paused_agents 移除 cid；即便后续子图抛出 CancelledError 也不会残留。"""
    import asyncio

    cid = await _cid(db_session, "t8-m4-leak")
    llm = _TicketModel(
        steps=[
            ("create_ticket", {"description": "耳机没声音", "ticket_type": "售后"}),
            "已创建",
        ]
    )
    graph = _parent(session_factory, llm)
    config = {"configurable": {"thread_id": f"t8-m4-{cid}"}}
    await graph.ainvoke(
        {"query": "耳机没声音，帮我建个售后工单", "conversation_id": cid, "messages": []},
        config=config,
    )
    assert cid in paused_agents

    class _CancelledAgent:
        async def ainvoke(self, *args, **kwargs):
            raise asyncio.CancelledError("client disconnected")

    paused_agents[cid]["agent"] = _CancelledAgent()
    with pytest.raises(BaseException):
        await graph.ainvoke(Command(resume={"action": "confirm"}), config=config)

    assert cid not in paused_agents


