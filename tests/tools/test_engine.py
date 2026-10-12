import asyncio
import logging

from sqlalchemy import select

from app.core.config import Settings
from app.db.models import ToolAuditLog
from app.tools.engine import execute
from app.tools.errors import LookupMiss
from app.tools.spec import ToolContext, ToolSpec

ORDER_SCHEMA = {
    "type": "object",
    "properties": {"order_no": {"type": "string", "minLength": 1}},
    "required": ["order_no"],
    "additionalProperties": False,
}

# 比超时大一个数量级，wait_for 取消后测试仍很快。
_SLOW = 0.4


def _settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        tool_timeout=0.05,
        tool_retries=1,
        tool_delay_tools="",
        tool_delay_seconds=0,
    )
    base.update(overrides)
    return Settings(**base)


def _ctx(**overrides) -> ToolContext:
    base = dict(conversation_id=7, tool_call_id="call-7", confirmed=False)
    base.update(overrides)
    return ToolContext(**base)


def _spec(handler, **overrides) -> ToolSpec:
    base = dict(
        name="query_order",
        description="查订单",
        parameters=ORDER_SCHEMA,
        permission="read",
        source="builtin",
        mcp_server=None,
        handler=handler,
        present=None,
    )
    base.update(overrides)
    return ToolSpec(**base)


async def _audit_rows(session_factory) -> list[dict]:
    async with session_factory() as session:
        rows = (
            await session.execute(select(ToolAuditLog).order_by(ToolAuditLog.id))
        ).scalars().all()
        return [
            {
                "conversation_id": row.conversation_id,
                "tool_call_id": row.tool_call_id,
                "tool_name": row.tool_name,
                "tool_source": row.tool_source,
                "mcp_server": row.mcp_server,
                "arguments": row.arguments,
                "result_summary": row.result_summary,
                "status": row.status,
                "error_message": row.error_message,
                "retry_count": row.retry_count,
                "duration_ms": row.duration_ms,
            }
            for row in rows
        ]


async def test_read_timeout_retries_once(db_session, session_factory):
    calls = {"n": 0}

    async def handler(args, ctx):
        calls["n"] += 1
        await asyncio.sleep(_SLOW)
        return {"order_no": args["order_no"]}

    result = await execute(
        _spec(handler),
        {"order_no": "1001"},
        _ctx(),
        session_factory=session_factory,
        settings=_settings(),
    )

    assert calls["n"] == 2
    assert result == {"ok": False, "error_type": "fault", "error": "工具执行超时"}
    rows = await _audit_rows(session_factory)
    assert len(rows) == 1
    assert rows[0]["status"] == "超时"
    assert rows[0]["retry_count"] == 1
    assert rows[0]["duration_ms"] is not None
    assert rows[0]["error_message"] == "工具执行超时"


async def test_read_timeout_then_success_records_one_retry(db_session, session_factory):
    calls = {"n": 0}

    async def handler(args, ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.sleep(_SLOW)
        return {"order_no": "1001", "_private": "hide"}

    result = await execute(
        _spec(handler),
        {"order_no": "1001"},
        _ctx(),
        session_factory=session_factory,
        settings=_settings(),
    )

    assert calls["n"] == 2
    assert result == {"ok": True, "data": {"order_no": "1001"}}
    rows = await _audit_rows(session_factory)
    assert rows[0]["status"] == "成功"
    assert rows[0]["retry_count"] == 1


async def test_write_timeout_does_not_retry(db_session, session_factory):
    calls = {"n": 0}

    async def handler(args, ctx):
        calls["n"] += 1
        await asyncio.sleep(_SLOW)
        return {"ticket_no": "T1"}

    result = await execute(
        _spec(handler, name="create_ticket", permission="write"),
        {"order_no": "1001"},
        _ctx(confirmed=True),
        session_factory=session_factory,
        settings=_settings(tool_retries=1),
    )

    assert calls["n"] == 1
    assert result == {"ok": False, "error_type": "fault", "error": "工具执行超时"}
    rows = await _audit_rows(session_factory)
    assert rows[0]["status"] == "超时"
    assert rows[0]["retry_count"] == 0


async def test_found_false_is_success_without_retry(db_session, session_factory):
    calls = {"n": 0}

    async def handler(args, ctx):
        calls["n"] += 1
        return {"found": False, "order_no": "1001", "_private": "hide"}

    result = await execute(
        _spec(handler),
        {"order_no": "1001"},
        _ctx(),
        session_factory=session_factory,
        settings=_settings(),
    )

    assert calls["n"] == 1
    assert result == {
        "ok": False,
        "error_type": "not_found",
        "error": "没有查到对应记录",
    }
    rows = await _audit_rows(session_factory)
    assert rows[0]["status"] == "成功"
    assert rows[0]["retry_count"] == 0
    assert "未查到" in rows[0]["result_summary"]
    assert "_private" not in rows[0]["result_summary"]


async def test_lookup_miss_is_not_retried(db_session, session_factory):
    calls = {"n": 0}

    async def handler(args, ctx):
        calls["n"] += 1
        raise LookupMiss("empty")

    result = await execute(
        _spec(handler),
        {"order_no": "1001"},
        _ctx(),
        session_factory=session_factory,
        settings=_settings(),
    )

    assert calls["n"] == 1
    assert result["error_type"] == "not_found"
    assert result["error"] == "没有查到对应记录"
    rows = await _audit_rows(session_factory)
    assert rows[0]["status"] == "成功"
    assert rows[0]["retry_count"] == 0
    assert "未查到" in rows[0]["result_summary"]


async def test_validation_and_unconfirmed_write_skip_handler(db_session, session_factory):
    calls = {"n": 0}

    async def handler(args, ctx):
        calls["n"] += 1
        return {"ticket_no": "T1"}

    invalid = await execute(
        _spec(handler, permission="read"),
        {},
        _ctx(),
        session_factory=session_factory,
        settings=_settings(),
    )
    denied = await execute(
        _spec(handler, name="create_ticket", permission="write"),
        {"order_no": "1001"},
        _ctx(confirmed=False),
        session_factory=session_factory,
        settings=_settings(),
    )

    assert calls["n"] == 0
    assert invalid == {
        "ok": False,
        "error_type": "invalid_args",
        "error": "参数不合法：缺少必填字段 order_no",
    }
    assert denied == {
        "ok": False,
        "error_type": "fault",
        "error": "写操作需要用户确认",
    }
    rows = await _audit_rows(session_factory)
    by_status = {row["status"]: row for row in rows}
    assert set(by_status) == {"校验拦下", "权限拒绝"}
    assert by_status["校验拦下"]["error_message"] == invalid["error"]
    assert by_status["校验拦下"]["retry_count"] == 0
    assert by_status["权限拒绝"]["error_message"] == "写操作需要用户确认"
    assert by_status["权限拒绝"]["retry_count"] == 0
    assert by_status["校验拦下"]["duration_ms"] is not None
    assert by_status["权限拒绝"]["duration_ms"] is not None


async def test_other_exception_is_fault_without_retry(db_session, session_factory):
    calls = {"n": 0}

    async def handler(args, ctx):
        calls["n"] += 1
        raise RuntimeError("数据库炸了")

    result = await execute(
        _spec(handler),
        {"order_no": "1001"},
        _ctx(),
        session_factory=session_factory,
        settings=_settings(),
    )

    assert calls["n"] == 1
    assert result["ok"] is False
    assert result["error_type"] == "fault"
    assert result["error"].startswith("工具执行失败：")
    rows = await _audit_rows(session_factory)
    assert rows[0]["status"] == "失败"
    assert rows[0]["retry_count"] == 0
    assert rows[0]["error_message"].startswith("工具执行失败：")


async def test_wrapped_connect_error_retries(db_session, session_factory):
    calls = {"n": 0}

    class ConnectError(Exception):
        pass

    async def handler(args, ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            try:
                raise ConnectError("reset")
            except ConnectError as inner:
                raise RuntimeError("wrapped") from inner
        return {"order_no": "1001"}

    result = await execute(
        _spec(handler),
        {"order_no": "1001"},
        _ctx(),
        session_factory=session_factory,
        settings=_settings(),
    )

    assert calls["n"] == 2
    assert result == {"ok": True, "data": {"order_no": "1001"}}
    rows = await _audit_rows(session_factory)
    assert rows[0]["status"] == "成功"
    assert rows[0]["retry_count"] == 1


async def test_delay_inside_attempt_can_time_out(db_session, session_factory):
    calls = {"n": 0}

    async def handler(args, ctx):
        calls["n"] += 1
        return {"order_no": "1001"}

    result = await execute(
        _spec(handler, name="query_order"),
        {"order_no": "1001"},
        _ctx(),
        session_factory=session_factory,
        settings=_settings(
            tool_retries=0,
            tool_timeout=0.05,
            tool_delay_tools="other, query_order",
            tool_delay_seconds=0.3,
        ),
    )

    assert calls["n"] == 0
    assert result["error"] == "工具执行超时"
    rows = await _audit_rows(session_factory)
    assert rows[0]["status"] == "超时"
    assert rows[0]["retry_count"] == 0


async def test_success_drops_private_keys_and_present_renames(db_session, session_factory):
    def handler(args, ctx):
        return {"order_no": args["order_no"], "_private": "secret", "status": "已发货"}

    ctx = _ctx()
    args = {"order_no": "1001"}
    spec = _spec(handler, source="mcp", mcp_server="logistics")
    dropped = await execute(
        spec,
        args,
        ctx,
        session_factory=session_factory,
        settings=_settings(tool_delay_tools="query_order", tool_delay_seconds=0),
    )
    assert dropped == {
        "ok": True,
        "data": {"order_no": "1001", "status": "已发货"},
    }
    assert isinstance(dropped, dict)

    def present(value):
        return {"订单号": value["order_no"]}

    renamed = await execute(
        _spec(handler, present=present),
        args,
        ctx,
        session_factory=session_factory,
        settings=_settings(),
    )
    assert renamed == {"ok": True, "data": {"订单号": "1001"}}

    rows = await _audit_rows(session_factory)
    assert len(rows) == 2
    first, second = rows
    assert first["status"] == "成功"
    assert first["retry_count"] == 0
    assert first["conversation_id"] == 7
    assert first["tool_call_id"] == "call-7"
    assert first["tool_name"] == "query_order"
    assert first["tool_source"] == "mcp"
    assert first["mcp_server"] == "logistics"
    assert first["arguments"] == {"order_no": "1001"}
    assert "_private" not in first["result_summary"]
    assert "已发货" in first["result_summary"]
    assert "\\u" not in first["result_summary"]
    assert first["duration_ms"] is not None
    assert "订单号" in second["result_summary"]
    assert "\\u" not in second["result_summary"]


async def test_audit_failure_still_returns_payload(db_session, session_factory, monkeypatch, caplog):
    from app.repositories.audit import AuditRepo

    called = {"n": 0}

    async def _boom(self, **kwargs):
        called["n"] += 1
        raise RuntimeError("audit down")

    monkeypatch.setattr(AuditRepo, "record", _boom)

    async def handler(args, ctx):
        return {"order_no": "1001", "_private": "secret"}

    with caplog.at_level(logging.ERROR):
        result = await execute(
            _spec(handler),
            {"order_no": "1001"},
            _ctx(),
            session_factory=session_factory,
            settings=_settings(),
        )

    assert called["n"] == 1
    assert result == {"ok": True, "data": {"order_no": "1001"}}
    assert any(record.levelno >= logging.ERROR for record in caplog.records)
