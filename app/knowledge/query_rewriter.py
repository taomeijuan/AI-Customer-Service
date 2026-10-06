import asyncio
import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field
from langchain.messages import HumanMessage

from app.extraction.service import build_structured_model

logger = logging.getLogger(__name__)

REWRITE_PROMPT = (
    "把用户问题改写归一成标准问法，并给出2-3个同义/近义扩展词（用于检索，不改变原意）。"
    '只输出 JSON：{"normalized": "...", "synonyms": ["..."]}'
)


class RewriteSchema(BaseModel):
    """Query 理解输出：归一问法 + 检索侧同义词扩展（只扩展检索词，不拆存入库）。"""

    normalized: str = Field(description="归一后的标准问法")
    synonyms: list[str] = Field(default_factory=list, description="同义/近义扩展词")


class QueryRewriter:
    """httpx 直连形态（测试用 MockTransport 注入）。失败降级为原话直用，绝不阻断检索链。"""

    def __init__(self, client: httpx.AsyncClient, model: str, retries: int = 1) -> None:
        self._client = client
        self.model = model
        self.retries = retries

    async def rewrite(self, query: str) -> RewriteSchema:
        prompt = f"{REWRITE_PROMPT}\n用户问题：{query}"
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = await self._client.post(
                    "/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return RewriteSchema.model_validate_json(content)
            except Exception as e:
                last_err = e
                logger.warning("rewrite attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(0.3 * (attempt + 1))
        logger.info("rewrite fallback to raw query")
        return RewriteSchema(normalized=query, synonyms=[])


class LangChainRewriter:
    """生产形态：走 LangChain 结构化输出（function_calling），失败同样降级原话。"""

    def __init__(self, llm: Any) -> None:
        self._structured = build_structured_model(llm, RewriteSchema)

    async def rewrite(self, query: str) -> RewriteSchema:
        try:
            result = await self._structured.ainvoke(
                [HumanMessage(f"{REWRITE_PROMPT}\n\n用户问题：{query}")]
            )
            if result["parsing_error"] or result["parsed"] is None:
                raise ValueError(str(result["parsing_error"]))
            return result["parsed"]
        except Exception as e:
            logger.warning("rewrite failed, fallback to raw: %s", e)
            return RewriteSchema(normalized=query, synonyms=[])


def build_query_rewriter(llm: Any) -> LangChainRewriter:
    return LangChainRewriter(llm)
