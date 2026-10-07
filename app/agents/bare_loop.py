"""祛魅裸 Agent 循环——不用任何框架，手写 OpenAI 协议的最裸 Agent。

看完这个文件你会发现：所谓 Agent 循环，本质就是——

    while True:
        回复 = 调一次 LLM(带工具定义)
        if 回复要调工具:
            结果 = 本地执行工具()
            把结果作为 tool 消息塞回对话
            continue          # 带着工具结果再问一次 LLM
        return 回复.content   # 模型认为够了，收敛出答案

框架（LangGraph 的 create_react_agent）帮你省掉的只有三件事：
节点编排、状态与检查点、流式事件。循环本身就这么大点事。
"""

import json
import logging

import httpx

logger = logging.getLogger(__name__)


def _tools_to_openai_schema(tools) -> list[dict]:
    """@tool 对象 → OpenAI functions 参数格式。"""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                # args_schema 是 pydantic 类（非实例）：导出 JSON Schema
                "parameters": t.args_schema.model_json_schema()
                if getattr(t, "args_schema", None)
                else {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


async def run_bare_loop(
    client: httpx.AsyncClient,
    model: str,
    tools: list,
    tool_executor,  # callable(name: str, args: dict) -> str
    messages: list[dict],
    max_steps: int = 6,
) -> str:
    """最裸的 Agent 循环。

    messages: OpenAI 格式的对话历史（含本轮用户消息）。
    max_steps: 最多执行几轮工具调用（防止模型停不下来）。
    返回：模型最终收敛出的文本答案。
    """
    convo = list(messages)
    schema = _tools_to_openai_schema(tools)
    last_text = ""  # 超步兜底：循环中每次 LLM 产出的非空文本

    for step in range(max_steps):
        resp = await client.post(
            "/chat/completions",
            json={"model": model, "messages": convo, "tools": schema},
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:  # 模型不再要工具 → 收敛出答案
            return msg.get("content") or ""

        last_text = msg.get("content") or last_text  # 记录已有最佳文本（超步兜底）
        convo.append(msg)  # 把「要工具」的 assistant 消息记进对话
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")
            result = tool_executor(name, args)  # 本地执行工具
            convo.append(
                {  # 结果作为 tool 消息喂回去——下一轮模型就能看到
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )

    logger.warning("bare loop hit max_steps=%s, returning last text", max_steps)
    return last_text  # 停止条件：超步时给已有最佳文本（非工具 JSON）
