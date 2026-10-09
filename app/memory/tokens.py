"""ch07 统一 token 口径：中文按字数折算（全仓唯一计数函数）。

需求校准点：**计数与预算必须同尺，只改一边净效果是反的**——
预算预留按本口径推导，trimmer/层预算/触发/日志全部走 estimate_tokens。

系数 2026-10-09 用真实语料（messages 120 条 + 知识答案 60 条，160 条含
CJK 样本）对 tiktoken cl100k 校准：tiktoken/估算 比值 p50=1.16、p90=1.25、
p95=1.33、max=1.67 → CJK 权重取 1.4（覆盖 p95 留余量，宁多勿少：
高估=提前降级、成本低；低估=爆窗口）。非 CJK 每 4 字符≈1 token。
"""

import math
import re

from langchain_core.messages import BaseMessage

# CJK 统一表意文字 + 全角标点/片假名区段
_CJK = re.compile(r"[㐀-䶿一-鿿 -〿＀-￯぀-ヿ]")
_CJK_WEIGHT = 1.4
_NONCJK_DIV = 4


def estimate_tokens(text: str) -> int:
    """统一口径：CJK 字 1.4 token/字 + 其余字符 ⌈/4⌉，至少 0。"""
    if not text:
        return 0
    n_cjk = len(_CJK.findall(text))
    n_other = len(text) - n_cjk
    return math.ceil(_CJK_WEIGHT * n_cjk + n_other / _NONCJK_DIV)


def count_message_tokens(message: BaseMessage) -> int:
    content = message.content if isinstance(message.content, str) else str(message.content)
    base = estimate_tokens(content)
    # 工具申请单的 name/args 也占上下文
    if getattr(message, "tool_calls", None):
        base += estimate_tokens(str(message.tool_calls))
    return base
