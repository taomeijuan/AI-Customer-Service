"""ch06 指代消解 + Query 改写节点（resolve 正式版，含订单槽位判断）。

用 LLM 结合对话历史，把「它能退吗」这类带指代/口语的消息补全成一句
不依赖上下文也能看懂的完整问题；问题已完整、指代已明确的**原样透传**。
失败（解析失败/超时）一律透传原 query（分流不能因消解挂掉而断）。

订单槽位 order_no（用户拍板规则：问句直通、动作要确认）：追问正在讨论
的那个订单时填入该单号；发起退款动作却没点名订单时留空，交给订单选择器
让用户自己确认。只能填历史里真实出现过的单号。
"""

import logging
from dataclasses import dataclass
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.workflow.text import CITE_RE, strip_stale_citations

logger = logging.getLogger(__name__)


class ResolutionSchema(BaseModel):
    """指代消解输出：补全问题 + 订单槽位（只填历史真实出现过的单号）。"""

    query: str = Field(description="改写后的完整问题；原句已完整时为原句")
    order_no: str = Field(default="", description="本消息指向的订单号，拿不准或动作未点名时留空")


@dataclass
class ResolveOutcome:
    query: str
    order_no: str


RESOLVE_SYSTEM_PROMPT = """你是客服系统的查询改写器。结合对话历史完成两件事：把用户最新消息改写成不依赖上下文也能看懂的完整问题，并判断这条消息指向哪个订单。

改写规则：
1. 指代消解：「它/这个/那个/这种」等指代词，必须根据历史替换成具体对象（商品名/订单号/政策名）。
2. 口语归一：模糊口语问法改成标准问法（如「咋退」→「怎么申请退货退款」），但不得改变用户本意。
3. 原样透传：问题已完整、指代已明确时，一字不改地返回原句；禁止润色、禁止增删信息、禁止回答问题本身。
4. 历史里找不到所指对象、无法消解时，原样返回，禁止瞎猜补全。

order_no 判定规则（只能填对话历史里真实出现过的订单号，绝不允许编造）：
- 用户在**追问刚才正在讨论的那个订单**（例：刚聊过订单1001，接着问「可以退货吗」「运费谁出」「它到哪了」）→ 填那个订单号
- 用户在**发起动作**（例：「我要退款」「帮我退掉」「申请退货」）却没有说出订单号 → 一律留空，由系统弹订单选择器让用户亲自确认
- 拿不准、话题已切换、历史里有多个订单竞争 → 一律留空

只输出 JSON：{"query": "<改写后的完整问题>", "order_no": "<订单号或空>"}"""


class Resolver:
    """resolve 节点实现：structured model（include_raw）→ 解析失败透传。"""

    def __init__(self, structured: Any) -> None:
        self._structured = structured  # build_structured_model(model, ResolutionSchema)

    async def resolve_detail(self, query: str, history: list) -> ResolveOutcome:
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
            return ResolveOutcome(query=query, order_no="")
        parsed = (out or {}).get("parsed") if isinstance(out, dict) else None
        resolved = str(
            (parsed.get("query") if isinstance(parsed, dict) else getattr(parsed, "query", "")) or ""
        ).strip()
        order_no = str(
            (parsed.get("order_no") if isinstance(parsed, dict) else getattr(parsed, "order_no", "")) or ""
        ).strip()
        if not resolved:
            return ResolveOutcome(query=query, order_no="")  # 解析失败/空结果 → 透传
        logger.info("resolve: %r -> %r (order_no=%r)", query, resolved, order_no)
        return ResolveOutcome(query=resolved, order_no=order_no)

    async def resolve(self, query: str, history: list) -> str:
        """兼容壳：只要改写句（旧调用/旧测试用）。"""
        return (await self.resolve_detail(query, history)).query
