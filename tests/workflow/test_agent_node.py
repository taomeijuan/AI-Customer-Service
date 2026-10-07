"""ReAct Agent 子图测试：create_react_agent 接线、多步工具、知识注入、步数上限。"""
import pytest
from langchain.messages import AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from app.tools.ecommerce import query_logistics, query_order

from app.workflow.agent_node import build_agent_node


class ScriptedChatModel(BaseChatModel):
    """脚本化 LLM（真正的 Runnable）：按「已见 ToolMessage 数」自适应取步。

    create_react_agent 要求 model 是 Runnable 且支持 bind_tools——裸对象会被拒。
    """

    steps: list  # 每步：AIMessage 文本 或 (工具名, 参数) 元组
    calls: int = 0
    bound: list = []
    invocations: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def bind_tools(self, tools, **kwargs):
        self.bound = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        self.invocations.append(list(messages))
        tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))
        step = self.steps[min(tool_count, len(self.steps) - 1)]
        if isinstance(step, tuple):  # (工具名, 参数) → 要工具
            name, args = step
            msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": name, "args": args, "id": f"c{tool_count + 1}", "type": "tool_call"}
                ],
            )
        else:
            msg = AIMessage(content=step)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _settings(max_steps=8):
    from types import SimpleNamespace

    return SimpleNamespace(agent_max_steps=max_steps)


def _make_llm(script):
    return ScriptedChatModel(steps=script)


def _node(llm, settings, session_factory):
    return build_agent_node(
        llm=llm,
        tools=[query_order, query_logistics],
        settings=settings,
        checkpointer=InMemorySaver(),
    )



@pytest.mark.usefixtures("db_session")
async def test_react_multi_step_two_tools(session_factory, db_session):
    """验收5：先查订单再查物流的复杂问题 → ReAct 走不止一步。"""
    llm = _make_llm(
        [
            (("query_order", {"order_no": "1001"})),
            (("query_logistics", {"order_no": "1001"})),
            "final",  # 收敛文本
        ]
    )
    node = _node(llm, _settings(8), session_factory)
    out = await node({"query": "订单1001发货了吗物流到哪了", "messages": [], "conversation_id": 1, "turn": 1})

    assert llm.calls == 3  # 两次工具 + 一次收敛
    assert llm.bound == ["query_order", "query_logistics"]  # 工具已绑定
    assert out["final_text"] == "final"
    # 两次 ToolMessage 确实进入了后续模型调用（工具结果被喂回）
    inv2_tool_msgs = [m for m in llm.invocations[1] if isinstance(m, ToolMessage)]
    assert len(inv2_tool_msgs) == 1
    inv3_tool_msgs = [m for m in llm.invocations[2] if isinstance(m, ToolMessage)]
    assert len(inv3_tool_msgs) == 2


@pytest.mark.usefixtures("db_session")
async def test_knowledge_injection(session_factory, db_session):
    """知识类路径：[n] 编号知识条目以 SystemMessage 注入 Agent。"""
    llm = _make_llm(["按[1]的政策回答"])
    node = _node(llm, _settings(6), session_factory)
    knowledge_msg = {"role": "system", "content": "已检索到相关知识：\n[1] 退货政策：7天无理由"}
    out = await node({"query": "退货政策", "messages": [knowledge_msg], "conversation_id": 1, "turn": 1})

    sent = llm.invocations[0]
    system_like = [m for m in sent if "7天无理由" in str(getattr(m, "content", ""))]
    assert system_like  # 知识注入可见
    assert out["final_text"] == "按[1]的政策回答"


@pytest.mark.usefixtures("db_session")
async def test_recursion_limit_guard(session_factory, db_session):
    """步数上限：模型永远要工具 → recursion_limit 兜住，返回兜底文本不炸会话。"""
    llm = _make_llm([(("query_order", {"order_no": "1001"}))] * 50)
    node = _node(llm, _settings(4), session_factory)
    out = await node({"query": "一直查", "messages": [], "conversation_id": 1, "turn": 1})
    assert "final_text" in out  # GraphRecursionError 被兜住
    assert llm.calls < 10  # 没有无限循环
