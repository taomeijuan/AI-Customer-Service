from langchain.messages import AIMessage, HumanMessage

from app.prompts.templates import SYSTEM_PROMPT, build_chat_prompt


def test_renders_system_history_human():
    prompt = build_chat_prompt()
    msgs = prompt.format_messages(
        history=[HumanMessage("之前说过什么"), AIMessage("说过订单")], input="帮我查订单"
    )
    assert msgs[0].type == "system" and "电商" in msgs[0].content
    assert msgs[1].content == "之前说过什么"
    assert msgs[2].type == "ai"
    assert msgs[-1].type == "human" and msgs[-1].content == "帮我查订单"


def test_empty_history_renders_system_human_only():
    msgs = build_chat_prompt().format_messages(history=[], input="你好")
    assert [m.type for m in msgs] == ["system", "human"]


def test_system_prompt_has_constraints():
    # 行为约束关键词在文案里（防文案回退）；文案本身按工作要求走人工核对
    # ①「对话记忆可用」与「实时数据查不到」分开表述（ch01 验收2 根因修复）
    # ②「政策费用类必须先查知识库」（ch02 验收3 漏召回演示的前提）
    # ③「明确建单才调 create_ticket」（投诉先安抚的产品决策）
    for kw in ("直接引用", "查不到", "query_faq", "create_ticket", "电商"):
        assert kw in SYSTEM_PROMPT
