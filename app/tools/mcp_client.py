"""按 Server 现拉 MCP 工具进注册表。权限一律只读，同名内置工具丢弃。"""

import json
import logging

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.tools import present as present_mod
from app.tools.spec import ToolSpec

logger = logging.getLogger(__name__)

_DEFAULT_SERVERS = ("logistics", "aftersales")
_EMPTY_SCHEMA = {"type": "object", "properties": {}}


def build_mcp_client(settings) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "logistics": {"transport": "http", "url": settings.mcp_logistics_url},
            "aftersales": {"transport": "http", "url": settings.mcp_aftersales_url},
        },
        tool_name_prefix=False,
        handle_tool_errors=False,
    )


def _parameters(tool) -> dict:
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return dict(_EMPTY_SCHEMA)
    # 适配器给的是 JSON Schema dict；自己构造的 LangChain 工具可能是 Pydantic 模型。
    if isinstance(schema, dict):
        return schema
    model_json_schema = getattr(schema, "model_json_schema", None)
    if callable(model_json_schema):
        return model_json_schema()
    return dict(_EMPTY_SCHEMA)


def _presenter(name: str):
    fn = getattr(present_mod, name, None)
    if callable(fn):
        return fn
    return None


def _coerce_tool_result(result: object) -> dict:
    """适配器把工具 dict 收成 text JSON。content block 的 id 每次都变，不能回给引擎。"""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        parsed = json.loads(result)
    elif isinstance(result, list):
        texts = []
        for block in result:
            if isinstance(block, dict):
                text = block.get("text")
            else:
                text = getattr(block, "text", None)
            if isinstance(text, str):
                texts.append(text)
        parsed = json.loads("".join(texts))
    else:
        raise TypeError(f"无法把 MCP 工具结果转成 dict：{type(result).__name__}")
    if not isinstance(parsed, dict):
        raise TypeError("MCP 工具结果不是 JSON 对象")
    return parsed


def _handler_for(tool):
    async def handler(args, ctx) -> dict:
        return _coerce_tool_result(await tool.ainvoke(args))

    return handler


def _drop_server(registry, server_name: str) -> None:
    """只删这个 Server 上次拉到的 MCP 工具。内置和其他 Server 留着。"""
    stale = [
        name
        for name, spec in registry._specs.items()
        if spec.source == "mcp" and spec.mcp_server == server_name
    ]
    for name in stale:
        del registry._specs[name]


async def refresh(
    registry,
    client,
    servers: tuple[str, ...] = _DEFAULT_SERVERS,
) -> None:
    """对每个 Server 现拉工具。连不上就跳过，不清空已有工具。"""
    for name in servers:
        try:
            tools = await client.get_tools(server_name=name)
        except Exception:
            logger.exception("拉取 MCP 工具失败 server=%s", name)
            continue
        _drop_server(registry, name)
        for tool in tools:
            existing = registry.get(tool.name)
            if existing is not None and existing.source == "builtin":
                logger.warning(
                    "跳过与内置工具同名的 MCP 工具 %s server=%s",
                    tool.name,
                    name,
                )
                continue
            registry.register(
                ToolSpec(
                    name=tool.name,
                    description=getattr(tool, "description", "") or "",
                    parameters=_parameters(tool),
                    permission="read",
                    source="mcp",
                    mcp_server=name,
                    handler=_handler_for(tool),
                    present=_presenter(tool.name),
                )
            )
