"""T5：create_ticket 未确认不落库，确认后才写 tickets。"""

import datetime
import re

from sqlalchemy import func, select

from app.core.config import Settings
from app.db.models import Ticket, ToolAuditLog
from app.repositories.conversations import ConversationsRepo
from app.tools.engine import execute
from app.tools.registry import ToolRegistry, scan_plugins
from app.tools.spec import ToolContext


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        tool_timeout=3.0,
        tool_retries=1,
        tool_delay_tools="",
        tool_delay_seconds=0,
    )


def _registry(session_factory) -> ToolRegistry:
    registry = ToolRegistry()
    registry.session_factory = session_factory
    scan_plugins(registry)
    return registry


async def _ticket_count(session_factory) -> int:
    async with session_factory() as session:
        count = await session.scalar(select(func.count()).select_from(Ticket))
        return int(count or 0)


async def _audit_statuses(session_factory) -> list[str]:
    async with session_factory() as session:
        rows = (
            await session.execute(select(ToolAuditLog.status).order_by(ToolAuditLog.id))
        ).scalars().all()
        return list(rows)


async def _conversation_id(db_session) -> int:
    return await ConversationsRepo(db_session).ensure_conversation(user_id="ticket-gate")


async def test_unconfirmed_write_does_not_insert(db_session, session_factory):
    cid = await _conversation_id(db_session)
    before = await _ticket_count(session_factory)
    spec = _registry(session_factory).get("create_ticket")
    result = await execute(
        spec,
        {"description": "屏幕碎了", "ticket_type": "售后"},
        ToolContext(conversation_id=cid, tool_call_id="call-deny", confirmed=False),
        session_factory=session_factory,
        settings=_settings(),
    )
    assert result["ok"] is False
    assert await _ticket_count(session_factory) == before
    assert await _audit_statuses(session_factory) == ["权限拒绝"]


async def test_confirmed_create_inserts_ticket(db_session, session_factory):
    cid = await _conversation_id(db_session)
    spec = _registry(session_factory).get("create_ticket")
    result = await execute(
        spec,
        {"description": "耳机没声音", "ticket_type": "售后"},
        ToolContext(conversation_id=cid, tool_call_id="call-ok", confirmed=True),
        session_factory=session_factory,
        settings=_settings(),
    )
    today = datetime.date.today().strftime("%Y%m%d")
    ticket_no = result["data"]["ticket_no"]
    assert result["ok"] is True
    assert re.fullmatch(rf"T{today}\d{{3}}", ticket_no)

    async with session_factory() as session:
        row = await session.get(Ticket, ticket_no)
    assert row is not None
    assert row.conversation_id == cid
    assert row.description == "耳机没声音"
    assert row.ticket_type == "售后"
    assert await _audit_statuses(session_factory) == ["成功"]


async def test_invalid_args_do_not_insert(db_session, session_factory):
    cid = await _conversation_id(db_session)
    spec = _registry(session_factory).get("create_ticket")
    ctx = ToolContext(conversation_id=cid, tool_call_id="call-bad", confirmed=True)
    before = await _ticket_count(session_factory)

    empty = await execute(
        spec,
        {"description": "", "ticket_type": "售后"},
        ctx,
        session_factory=session_factory,
        settings=_settings(),
    )
    other = await execute(
        spec,
        {"description": "想问一下", "ticket_type": "其他"},
        ctx,
        session_factory=session_factory,
        settings=_settings(),
    )

    assert empty["ok"] is False
    assert other["ok"] is False
    assert await _ticket_count(session_factory) == before
    assert await _audit_statuses(session_factory) == ["校验拦下", "校验拦下"]
