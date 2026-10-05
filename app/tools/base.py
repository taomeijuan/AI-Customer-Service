import asyncio
import logging
from typing import Any

from pydantic import ValidationError

logger = logging.getLogger(__name__)


class ToolRegistry:
    """@tool 注册表：Pydantic 校验 → 超时 → 重试 → 异常包装 {ok, data|error}。

    工具执行永不抛异常——错误统一以 {"ok": False, "error": ...} 回灌给模型，
    由模型组织"抱歉/转人工"类回答，而不是炸掉会话。
    """

    def __init__(self, default_timeout: float = 3.0, default_retries: int = 1) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self.default_timeout = default_timeout
        self.default_retries = default_retries

    def register(
        self, tool_obj: Any, timeout: float | None = None, retries: int | None = None
    ) -> None:
        self._tools[tool_obj.name] = {
            "tool": tool_obj,
            "timeout": self.default_timeout if timeout is None else timeout,
            "retries": self.default_retries if retries is None else retries,
        }

    def all(self) -> list:
        return [e["tool"] for e in self._tools.values()]

    async def execute(self, name: str, args: dict) -> dict:
        entry = self._tools.get(name)
        if entry is None:
            return {"ok": False, "error": f"工具未注册: {name}"}
        tool_obj = entry["tool"]
        timeout, retries = entry["timeout"], entry["retries"]

        try:
            kwargs = tool_obj.args_schema.model_validate(args).model_dump()
        except ValidationError as e:
            return {"ok": False, "error": f"参数校验失败: {e.errors()[0]['msg']}"}
        except AttributeError:  # 无 args_schema 的工具按原样传参
            kwargs = args

        fn = tool_obj.coroutine or tool_obj.func
        last_err = "执行失败"
        for attempt in range(retries + 1):
            try:
                if asyncio.iscoroutinefunction(fn):
                    data = await asyncio.wait_for(fn(**kwargs), timeout=timeout)
                else:
                    data = await asyncio.wait_for(
                        asyncio.to_thread(fn, **kwargs), timeout=timeout
                    )
                return {"ok": True, "data": data}
            except asyncio.TimeoutError:
                last_err = f"执行超时（>{timeout}s）"
                logger.warning("tool %s attempt %d: %s", name, attempt + 1, last_err)
            except Exception as e:
                last_err = f"执行失败: {e}"
                logger.warning("tool %s attempt %d: %s", name, attempt + 1, last_err)
        return {"ok": False, "error": last_err}
