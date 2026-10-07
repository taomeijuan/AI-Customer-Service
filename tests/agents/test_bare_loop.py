"""祛魅裸循环测试：手写 OpenAI 协议 Agent 循环的三种命运。"""
import json

import httpx
import pytest

from app.agents.bare_loop import run_bare_loop
from app.tools.ecommerce import query_logistics, query_order


def _llm_response(content=None, tool_calls=None):
    """构造一条 OpenAI 格式的 assistant 消息。"""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(tool_calls)
        ]
    return msg


def make_loop(script):
    """script: 每轮 LLM 返回的 assistant 消息列表；同时记录工具执行轨迹。"""
    executed = []
    state = {"turn": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = min(state["turn"], len(script) - 1)
        state["turn"] += 1
        msg = script[i]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": _llm_response(
                            msg.get("content"),
                            [(n, a) for n, a in msg.get("tool_calls", [])],
                        )
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1")

    def tool_executor(name: str, args: dict) -> str:
        executed.append((name, args))
        return json.dumps({"mock": f"{name} 执行结果"})

    from functools import partial

    return (
        partial(
            run_bare_loop,
            client=client,
            model="test",
            tools=[query_order, query_logistics],
            tool_executor=tool_executor,
        ),
        executed,
    )


async def test_one_tool_call_then_converge():
    """第一轮要工具、执行喂回、第二轮收敛——Agent 循环的最小完整闭环。"""
    loop, executed = make_loop(
        [
            {"tool_calls": [("query_order", {"order_no": "1001"})]},
            {"content": "订单 1001 已发货"},
        ]
    )
    out = await loop(messages=[{"role": "user", "content": "订单1001呢"}])
    assert out == "订单 1001 已发货"
    assert executed == [("query_order", {"order_no": "1001"})]


async def test_multi_step_two_tools():
    """复杂问题：先查订单再查物流，两步工具后收敛。"""
    loop, executed = make_loop(
        [
            {"tool_calls": [("query_order", {"order_no": "1001"})]},
            {"tool_calls": [("query_logistics", {"order_no": "1001"})]},
            {"content": "该订单物流显示运输中"},
        ]
    )
    out = await loop(messages=[{"role": "user", "content": "订单1001发货了吗到哪了"}])
    assert out == "该订单物流显示运输中"
    assert [name for name, _ in executed] == ["query_order", "query_logistics"]


async def test_max_steps_stops():
    """每轮都要工具 → 跑满 max_steps 停止，不死循环。"""
    loop, executed = make_loop(
        [{"tool_calls": [("query_order", {"order_no": "1001"})]}] * 10
    )
    out = await loop(messages=[{"role": "user", "content": "一直查"}], max_steps=3)
    assert len(executed) == 3  # 恰好 max_steps 次工具执行后停止


async def test_no_tool_direct_answer():
    """首轮就无工具调用 → 一次 LLM 调用直接收敛，零工具执行。"""
    loop, executed = make_loop([{"content": "您好，请问有什么可以帮您"}])
    out = await loop(messages=[{"role": "user", "content": "你好"}])
    assert out == "您好，请问有什么可以帮您"
    assert executed == []
