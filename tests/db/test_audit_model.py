from sqlalchemy import select

from app.db.models import ToolAuditLog


def test_tool_audit_log_has_no_foreign_keys():
    assert not ToolAuditLog.__table__.foreign_keys


async def test_status_validation_blocked_roundtrip(db_session):
    row = ToolAuditLog(
        tool_name="query_order",
        tool_source="builtin",
        status="校验拦下",
    )
    db_session.add(row)
    await db_session.commit()
    loaded = (
        await db_session.execute(select(ToolAuditLog).where(ToolAuditLog.id == row.id))
    ).scalar_one()
    assert loaded.status == "校验拦下"
