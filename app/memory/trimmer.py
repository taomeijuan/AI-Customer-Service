import tiktoken
from langchain_core.messages import BaseMessage

_encoding = tiktoken.get_encoding("cl100k_base")  # 四家上游统一近似


def count_tokens(message: BaseMessage) -> int:
    return len(_encoding.encode(str(message.content)))


def trim_history(
    messages: list[BaseMessage], budget_tokens: int, min_keep: int = 1
) -> list[BaseMessage]:
    """从最新往回保留预算内的消息；system 不入会话历史，天然不参与裁剪。"""
    kept: list[BaseMessage] = []
    total = 0
    for msg in reversed(messages):
        cost = count_tokens(msg)
        if kept and total + cost > budget_tokens:  # min_keep：即使超预算也保留最近一轮
            break
        kept.append(msg)
        total += cost
    return list(reversed(kept))
