import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """检索证据：引用编号映射的最小单元。"""

    chunk_id: int
    section_path: str
    question: str
    answer: str
    score: float = 0.0


class Reranker:
    """SiliconFlow /rerank 客户端（bge-reranker-v2-m3）。失败降级为原序（score=0）。"""

    def __init__(self, client: httpx.AsyncClient, model: str, retries: int = 1) -> None:
        self._client = client
        self.model = model
        self.retries = retries

    async def rerank(self, query: str, evidences: list[Evidence], top_n: int) -> tuple[list[Evidence], bool]:
        """返回 (精排后证据, 是否精排成功)。

        失败降级为原序并把第二位返回 False——调用方（retriever）据此跳过置信度
        阈值判定：降级时 Evidence.score 仍是 RRF/BM25 分数，与 0-1 的精排分
        不可比，误比较会导致全量误判低置信。
        """
        documents = [e.answer for e in evidences]
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = await self._client.post(
                    "/rerank",
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": documents,
                        "top_n": top_n,
                        "return_documents": False,
                    },
                )
                resp.raise_for_status()
                out: list[Evidence] = []
                for item in resp.json()["results"]:
                    ev = evidences[item["index"]]
                    out.append(
                        Evidence(
                            chunk_id=ev.chunk_id,
                            section_path=ev.section_path,
                            question=ev.question,
                            answer=ev.answer,
                            score=item["relevance_score"],
                        )
                    )
                return out, True
            except Exception as e:
                last_err = e
                logger.warning("rerank attempt %d failed: %s", attempt + 1, e)
        logger.warning("rerank degraded to original order: %s", last_err)
        return evidences[:top_n], False


def build_reranker(settings: Settings) -> Reranker | None:
    """未配置 key 时返回 None，retriever 跳过精排。"""
    if not getattr(settings, "rerank_api_key", ""):
        return None
    client = httpx.AsyncClient(
        base_url=settings.rerank_api_base,
        headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
        timeout=30,
    )
    return Reranker(client=client, model=settings.rerank_model)
