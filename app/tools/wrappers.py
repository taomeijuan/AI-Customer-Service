"""把 ToolSpec 收成 LangChain 工具。建工单在写库前 interrupt，其余直接走引擎。"""

import json
from dataclasses import replace
from typing import Annotated, Any

from langchain_core.tools import InjectedToolCallId, StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field, create_model

from app.tools.engine import execute
from app.tools.spec import ToolContext, ToolSpec
from app.tools.validate import validate_args

_CANCELLED = "用户取消了建单"

_PY_TYPES = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _py_type(prop: dict) -> Any:
    raw = prop.get("type", "string")
    if isinstance(raw, list):
        raw = next((item for item in raw if item != "null"), "string")
    return _PY_TYPES.get(raw, Any)


def _args_model(spec: ToolSpec) -> type[BaseModel]:
    """JSON Schema 的字段名和类型 → Pydantic。长度/枚举留给 validate_args，避免空串进不了工具体。"""
    schema = spec.parameters or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            prop = {}
        description = prop.get("description") or ""
        if prop.get("enum"):
            allowed = "、".join(str(item) for item in prop["enum"])
            description = f"{description} 允许取值：{allowed}".strip()
        annotation = _py_type(prop)
        if name in required:
            fields[name] = (annotation, Field(..., description=description))
        else:
            fields[name] = (annotation | None, Field(default=None, description=description))
    fields["tool_call_id"] = (
        Annotated[str, InjectedToolCallId],
        Field(default="", description="工具调用 id"),
    )
    return create_model(f"{spec.name}_Args", **fields)


class _EngineTool(StructuredTool):
    """透传 dict 入参，把缺必填字段、类型错误、多余字段统一交给 validate_args 与引擎审计。"""

    def _parse_input(
        self,
        tool_input: str | dict,
        tool_call_id: str | None = None,
    ) -> str | dict:
        if isinstance(tool_input, dict):
            result = dict(tool_input)
            if tool_call_id is not None:
                result["tool_call_id"] = tool_call_id
            return result
        return super()._parse_input(tool_input, tool_call_id=tool_call_id)


def _context(ctx_factory, tool_call_id: str | None) -> ToolContext:
    ctx = ctx_factory()
    if tool_call_id:
        return replace(ctx, tool_call_id=tool_call_id)
    return ctx


def _tool(spec: ToolSpec, ctx_factory, *, session_factory, settings) -> StructuredTool:
    async def _run(**kwargs: Any) -> str:
        tool_call_id = kwargs.pop("tool_call_id", None) or None
        args = dict(kwargs)
        ctx = _context(ctx_factory, tool_call_id)
        if spec.name == "create_ticket":
            if validate_args(spec.parameters, args) is not None:
                result = await execute(
                    spec, args, ctx, session_factory=session_factory, settings=settings
                )
                return json.dumps(result, ensure_ascii=False)
            decision = interrupt(
                {
                    "type": "ticket_preview",
                    "ticket_type": args.get("ticket_type"),
                    "description": args.get("description"),
                }
            )
            if isinstance(decision, dict) and decision.get("action") == "confirm":
                ctx = replace(ctx, confirmed=True)
                result = await execute(
                    spec, args, ctx, session_factory=session_factory, settings=settings
                )
            else:
                result = await execute(
                    spec,
                    args,
                    ctx,
                    session_factory=session_factory,
                    settings=settings,
                    deny_message=_CANCELLED,
                )
            return json.dumps(result, ensure_ascii=False)
        result = await execute(
            spec, args, ctx, session_factory=session_factory, settings=settings
        )
        return json.dumps(result, ensure_ascii=False)

    return _EngineTool.from_function(
        coroutine=_run,
        name=spec.name,
        description=spec.description or spec.name,
        args_schema=_args_model(spec),
        infer_schema=False,
    )


def specs_to_tools(specs, ctx_factory, *, session_factory, settings) -> list:
    """每个 spec 一个 LangChain 工具。协程返回引擎结果的 JSON 字符串。"""
    return [
        _tool(spec, ctx_factory, session_factory=session_factory, settings=settings)
        for spec in specs
    ]
