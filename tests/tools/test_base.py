import asyncio

import pytest
from langchain.tools import tool

from app.tools.base import ToolRegistry


@tool
def add(a: int, b: int) -> dict:
    """加法。"""
    return {"sum": a + b}


@tool
def boom(x: str) -> dict:
    """总是炸。"""
    raise RuntimeError("上游挂了")


@tool
async def sleepy(x: str) -> dict:
    """睡1秒。"""
    await asyncio.sleep(1)
    return {}


def make_registry(timeout=0.2, retries=1) -> ToolRegistry:
    reg = ToolRegistry(default_timeout=timeout, default_retries=retries)
    reg.register(add)
    reg.register(boom)
    return reg


async def test_success():
    out = await make_registry().execute("add", {"a": 1, "b": 2})
    assert out == {"ok": True, "data": {"sum": 3}}


async def test_schema_validation():
    out = await make_registry().execute("add", {"a": "不是数字", "b": 2})
    assert out["ok"] is False and "参数校验失败" in out["error"]


async def test_unknown_tool():
    out = await make_registry().execute("nope", {})
    assert out["ok"] is False and "未注册" in out["error"]


async def test_error_wrapped():
    out = await make_registry().execute("boom", {"x": "1"})
    assert out["ok"] is False and "上游挂了" in out["error"]


async def test_timeout():
    reg = ToolRegistry(default_timeout=0.1, default_retries=0)
    reg.register(sleepy)
    out = await reg.execute("sleepy", {"x": "1"})
    assert out["ok"] is False and "超时" in out["error"]


async def test_retry_until_success():
    calls = {"n": 0}

    @tool
    def flaky(x: str) -> dict:
        """前两次失败。"""
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("抖动")
        return {"n": calls["n"]}

    reg = ToolRegistry(default_timeout=1.0, default_retries=2)
    reg.register(flaky)
    out = await reg.execute("flaky", {"x": "1"})
    assert out == {"ok": True, "data": {"n": 3}}  # 重试 2 次后第 3 次成功


async def test_list_for_bind_tools():
    reg = make_registry()
    assert [t.name for t in reg.all()] == ["add", "boom"]
