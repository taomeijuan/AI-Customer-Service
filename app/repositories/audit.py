from app.db.models import ToolAuditLog


class AuditRepo:
    """tool_audit_logs 插入。调用方开新会话并 commit。"""

    def __init__(self, session) -> None:
        self._session = session

    async def record(
        self,
        *,
        conversation_id: int | None,
        tool_call_id: str | None,
        tool_name: str,
        tool_source: str,
        mcp_server: str | None,
        arguments: dict | None,
        result_summary: str | None,
        status: str,
        error_message: str | None,
        retry_count: int,
        duration_ms: int | None,
    ) -> None:
        self._session.add(
            ToolAuditLog(
                conversation_id=conversation_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_source=tool_source,
                mcp_server=mcp_server,
                arguments=arguments,
                result_summary=result_summary,
                status=status,
                error_message=error_message,
                retry_count=retry_count,
                duration_ms=duration_ms,
            )
        )
