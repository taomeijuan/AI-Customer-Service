"""ch06 指代消解 + Query 改写节点（resolve 正式版）。

用 LLM 结合对话历史，把「它能退吗」这类带指代/口语的消息补全成一句
不依赖上下文也能看懂的完整问题；问题已完整、指代已明确的**原样透传**。
失败（解析失败/空结果）一律透传原句——分流不能因消解挂掉而断流。
"""

import logging
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.workflow.text import CITE_RE, strip_stale_citations

logger = logging.getLogger(__name__)


class ResolutionSchema(BaseModel):
    """指代消解输出：只补全不改意。"""

    query: str = Field(description="改写后的完整问题；原句已完整时为原句")


RESOLVE_SYSTEM_PROMPT = """你是客服系统的查询改写器。结合对话历史，把用户最新消息改写成一句不依赖上下文也能看懂的完整问题。

规则：
1. 指代消解：「它/这个/那个/这种」等指代词，必须根据历史替换成具体对象（商品名/订单号/政策名）。
2. 口语归一：模糊口语问法改成标准问法（如「咋退」→「怎么申请退货退款」、「多少钱」→「价格是多少」），但不得改变用户本意。
3. 原样透传：问题已完整、指代已明确时，一字不改地返回原句；禁止润色、禁止增删信息、禁止回答问题本身。
4. 历史里找不到所指对象、无法消解时，原样返回，禁止瞎猜补全。

只输出 JSON：{"query": "<改写后的完整问题>"}"""


class Resolver:
    """resolve 节点实现：structured model（include_raw）→ 解析失败透传。"""

    def __init__(self, structured: Any) -> None:
        self._structured = structured  # build_structured_model(model, ResolutionSchema)

    async def resolve(self, query: str, history: list) -> str:
        msgs: list = [SystemMessage(RESOLVE_SYSTEM_PROMPT)]
        for m in history or []:
            content = getattr(m, "content", "")
            if isinstance(content, str) and CITE_RE.search(content):
                m = m.model_copy(update={"content": strip_stale_citations(content)})
            msgs.append(m)
        msgs.append(HumanMessage(query))

        try:
            out = await self._structured.ainvoke(msgs)
        except Exception as e:
            logger.warning("resolve failed, pass-through: %s", e)
            return query
        parsed = (out or {}).get("parsed") if isinstance(out, dict) else None
        resolved = ""
        if isinstance(parsed, dict):
            resolved = str(parsed.get("query") or "")
        else:
            resolved = str(getattr(parsed, "query", "") or "")
        resolved = resolved.strip()
        if not resolved:
            return query  # 解析失败/空结果 → 透传
        logger.info("resolve: %r -> %r", query, resolved)
        return resolved
