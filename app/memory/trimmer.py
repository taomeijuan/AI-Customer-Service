import tiktoken
from langchain_core.messages import BaseMessage

_encoding = tiktoken.get_encoding("cl100k_base")  # 四家上游统一近似


def count_tokens(message: BaseMessage) -> int:
    return len(_encoding.encode(str(message.content)))


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
