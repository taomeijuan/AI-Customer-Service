from langchain.messages import HumanMessage
from langchain_core.messages import BaseMessage

from app.memory.tokens import count_message_tokens


def count_tokens(message: BaseMessage) -> int:
    """ch07：切统一字数折算口径（原 tiktoken 见 tokens.py 校准记录）。"""
    return count_message_tokens(message)


def trim_history(messages: list[BaseMessage], budget_tokens: int) -> list[BaseMessage]:
    """从最新往回保留预算内的消息。

    system 不入会话历史，天然不参与裁剪；历史始终按 [Human, AI] 成对回填，
    超预算时至少保留最近 1 轮（最后 2 条），避免留下孤儿回复。
    """
    kept: list[BaseMessage] = []
    total = 0
    for msg in reversed(messages):
        cost = count_tokens(msg)
        if kept and total + cost > budget_tokens:
            break
        kept.append(msg)
        total += cost
    if len(kept) < 2 and len(messages) >= 2:  # 至少保留最近一轮，优先于预算
        return list(messages[-2:])
    return list(reversed(kept))


def _group_rounds(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """按 HumanMessage 分轮：每轮 = user + 其后的 assistant/tool 往返。

    轮是原子裁剪单位——tool 结果与其配对的 assistant 申请单绝不能被裁开，
    否则回灌上游会报 400（tool 必须紧跟带 tool_calls 的消息）。
    """
    groups: list[list[BaseMessage]] = []
    for m in messages:
        if isinstance(m, HumanMessage) or not groups:
            groups.append([m])
        else:
            groups[-1].append(m)
    return groups


def trim_history_groups(
    messages: list[BaseMessage], budget_tokens: int
) -> list[BaseMessage]:
    """带工具往返的裁剪：按轮分组、整轮保留/丢弃；至少保留最近一轮。"""
    groups = _group_rounds(messages)
    kept: list[list[BaseMessage]] = []
    total = 0
    for group in reversed(groups):
        cost = sum(count_tokens(m) for m in group)
        if kept and total + cost > budget_tokens:
            break
        kept.insert(0, group)
        total += cost
    if not kept and groups:  # 至少保留最近一轮，优先于预算
        kept = [groups[-1]]
    return [m for group in kept for m in group]
