"""ch06 Query 扩写器：把一条退款/售后问题泛化成多条侧重点不同的检索查询。

只服务退款确定性子流程（高频、容错低的核心场景），检索侧现查现用，
知识库不拆存多份。失败兜底 → 返回原查询单路检索，不断流。
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

EXPAND_QUERIES_MIN = 2
EXPAND_QUERIES_MAX = 4


class ExpansionSchema(BaseModel):
    queries: list[str] = Field(
        min_length=1, max_length=5, description="2-4 条侧重点不同的检索查询"
    )


EXPAND_SYSTEM_PROMPT = """你是客服知识库的检索查询扩写器。把用户的一条退款/售后问题，扩写成 2-4 条侧重点不同、各自可独立检索的标准查询。

角度参考（按需选取，不必凑齐）：
- 政策条款角度：直接问对应规则（如「七天无理由退货条件」）
- 时效角度：问时限（如「退货申请时间限制」「退款到账时间」）
- 场景特例角度：问特殊情形（如「拆封商品能否退货」「生鲜退换规则」）

要求：
1. 每条查询必须独立成立，去掉「它/这个」等指代，补全对象
2. 不得改变用户本意，不得编造用户没提的场景
3. 每条不超过 30 字，口语归一为适合检索的书面表达

只输出 JSON：{"queries": ["<查询1>", "<查询2>", ...]}"""


class Expander:
    """扩写器：structured model（include_raw）→ 解析失败/空结果兜底原查询。"""

    def __init__(self, structured: Any) -> None:
        self._structured = structured  # build_structured_model(model, ExpansionSchema)

    async def expand(self, query: str, context: str | None = None) -> list[str]:
        """context：订单/商品摘要（可选），帮助扩出更贴合的查询。"""
        user_content = f"用户问题：{query}"
        if context:
            user_content += f"\n相关背景：{context}"

        try:
            out = await self._structured.ainvoke([user_content])
            parsed = (out or {}).get("parsed") if isinstance(out, dict) else None
            queries: list[str] = []
            raw_list = parsed.get("queries") if isinstance(parsed, dict) else getattr(parsed, "queries", None)
            for q in raw_list or []:
                q = str(q).strip()
                if q and q not in queries:
                    queries.append(q)
            if len(queries) < EXPAND_QUERIES_MIN:
                raise ValueError(f"扩写数量不足: {len(queries)}")
            return queries[:EXPAND_QUERIES_MAX]
        except Exception as e:
            logger.warning("expand failed, fallback to original query: %s", e)
            return [query]
