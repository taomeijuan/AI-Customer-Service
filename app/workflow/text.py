"""Workflow 共享文本处理：陈旧引用角标剥离（ch06 从 agent_node 抽出共用）。"""

import re

# 陈旧角标：历史 AI 回答里的 [n] 只属于当时的轮次，本轮没有对应证据。
# 不剥掉的话模型会把旧编号直接抄进新答案，前端拿到 citations 对不上 → 死文本。
CITE_RE = re.compile(r"\[\d+\]")


def strip_stale_citations(text: str) -> str:
    return CITE_RE.sub("", text)
