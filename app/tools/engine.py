import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass

from app.repositories.audit import AuditRepo
from app.tools.errors import LookupMiss
from app.tools.spec import ToolContext, ToolSpec
from app.tools.validate import validate_args

logger = logging.getLogger(__name__)

_SUMMARY_LIMIT = 2000
_ERROR_LIMIT = 512
_RETRY_NAME_PARTS = (
    "ConnectError",
    "ReadError",
    "RemoteProtocolError",
    "NetworkError",
    "PoolTimeout",
)
_NOT_FOUND = "没有查到对应记录"
_NEED_CONFIRM = "写操作需要用户确认"
_TIMEOUT = "工具执行超时"


@dataclass
class _Outcome:
    payload: dict
    status: str
    error_message: str | None
    result_summary: str | None
    retry_count: int = 0


def _summary(value, *, not_found: bool = False) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = str(value)
    if not_found:
        text = f"未查到 {text}"
    return text[:_SUMMARY_LIMIT]


def _clip(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    return text[:limit]


def _present(spec: ToolSpec, value):
    if spec.present is not None:
        return spec.present(value)
    if isinstance(value, dict):
        return {key: item for key, item in value.items() if not str(key).startswith("_")}
    return value


def _delay_seconds(spec: ToolSpec, settings) -> float:
    seconds = settings.tool_delay_seconds
    if seconds <= 0:
        return 0.0
    names = {part.strip() for part in settings.tool_delay_tools.split(",") if part.strip()}
    if spec.name not in names:
        return 0.0
    return seconds


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, asyncio.TimeoutError))


def _is_retryable(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
            return True
        name = type(current).__name__
        if any(part in name for part in _RETRY_NAME_PARTS):
            return True
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            break
    return False


def _ok(data, retry_count: int) -> _Outcome:
    return _Outcome(
        payload={"ok": True, "data": data},
        status="成功",
        error_message=None,
        result_summary=_summary(data),
        retry_count=retry_count,
    )


def _not_found(formatted, retry_count: int) -> _Outcome:
    return _Outcome(
        payload={"ok": False, "error_type": "not_found", "error": _NOT_FOUND},
        status="成功",
        error_message=None,
        result_summary=_summary(formatted, not_found=True),
        retry_count=retry_count,
    )


def _from_exception(exc: BaseException, retry_count: int) -> _Outcome:
    if _is_timeout(exc):
        return _Outcome(
            payload={"ok": False, "error_type": "fault", "error": _TIMEOUT},
            status="超时",
            error_message=_TIMEOUT,
            result_summary=_summary({"error": _TIMEOUT}),
            retry_count=retry_count,
        )
    message = f"工具执行失败：{exc}"
    return _Outcome(
        payload={"ok": False, "error_type": "fault", "error": message},
        status="失败",
        error_message=message,
        result_summary=_summary({"error": message}),
        retry_count=retry_count,
    )


async def _attempt(spec: ToolSpec, args: dict, ctx: ToolContext, settings):
    delay = _delay_seconds(spec, settings)
    if delay > 0:
        await asyncio.sleep(delay)
    handler = spec.handler
    if inspect.iscoroutinefunction(handler):
        return await handler(args, ctx)
    return await asyncio.to_thread(handler, args, ctx)


async def _run(spec: ToolSpec, args: dict, ctx: ToolContext, settings) -> _Outcome:
    retries = settings.tool_retries if spec.permission == "read" else 0
    if retries < 0:
        retries = 0
    extra = 0
    for attempt in range(retries + 1):
        try:
            value = await asyncio.wait_for(
                _attempt(spec, args, ctx, settings),
                timeout=settings.tool_timeout,
            )
        except LookupMiss:
            return _not_found({}, extra)
        except Exception as exc:
            if _is_retryable(exc) and attempt < retries:
                extra += 1
                continue
            return _from_exception(exc, extra)
        try:
            if isinstance(value, dict) and value.get("found") is False:
                return _not_found(_present(spec, value), extra)
            data = _present(spec, value)
        except Exception as exc:
            return _from_exception(exc, extra)
        return _ok(data, extra)
    return _from_exception(RuntimeError("工具没有返回结果"), extra)


async def _audit(session_factory, spec: ToolSpec, args: dict, ctx: ToolContext, outcome: _Outcome, duration_ms: int) -> None:
    try:
        async with session_factory() as session:
            await AuditRepo(session).record(
                conversation_id=ctx.conversation_id,
                tool_call_id=ctx.tool_call_id,
                tool_name=spec.name,
                tool_source=spec.source,
                mcp_server=spec.mcp_server,
                arguments=args,
                result_summary=outcome.result_summary,
                status=outcome.status,
                error_message=_clip(outcome.error_message, _ERROR_LIMIT),
                retry_count=outcome.retry_count,
                duration_ms=duration_ms,
            )
            await session.commit()
    except Exception:
        logger.exception("工具审计写入失败 tool=%s", spec.name)


async def execute(spec: ToolSpec, args: dict, ctx: ToolContext, *, session_factory, settings) -> dict:
    """校验 → 写操作确认门 → 调用（只读才重试）→ 格式化 → 审计。返回给模型的 dict。"""
    started = time.perf_counter()
    invalid = validate_args(spec.parameters, args)
    if invalid is not None:
        outcome = _Outcome(
            payload={"ok": False, "error_type": "invalid_args", "error": invalid},
            status="校验拦下",
            error_message=invalid,
            result_summary=_summary({"error": invalid}),
            retry_count=0,
        )
    elif spec.permission == "write" and not ctx.confirmed:
        outcome = _Outcome(
            payload={"ok": False, "error_type": "fault", "error": _NEED_CONFIRM},
            status="权限拒绝",
            error_message=_NEED_CONFIRM,
            result_summary=_summary({"error": _NEED_CONFIRM}),
            retry_count=0,
        )
    else:
        outcome = await _run(spec, args, ctx, settings)
    duration_ms = int((time.perf_counter() - started) * 1000)
    await _audit(session_factory, spec, args, ctx, outcome, duration_ms)
    return outcome.payload
