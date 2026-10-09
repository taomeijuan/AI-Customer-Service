import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from app.knowledge.reranker import Evidence, Reranker
from app.knowledge.query_rewriter import RewriteSchema

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果：证据列表 + 检索侧低置信标记（编排器据此走拒答路径）。"""

    evidences: list[Evidence] = field(default_factory=list)
    low_confidence: bool = False


class HybridRetriever:
    """混合检索编排：改写 → dense+BM25 融合 → 元数据过滤 → 精排。

    strategy: dense / bm25 / hybrid / hybrid_rerank（评估对比与降级用）。
    reranker 为 None 时 hybrid_rerank 自动退化为 hybrid。
    """

    def __init__(
        self,
        milvus: Any,
        embedder: Any,
        rewriter: Any,
        reranker: Reranker | None,
        settings: Any,
    ) -> None:
        self._milvus = milvus
        self._embedder = embedder
        self._rewriter = rewriter
        self._reranker = reranker
        self._settings = settings

    async def retrieve(
        self,
        query: str,
        strategy: str = "hybrid_rerank",
        category_prefix: str | None = None,
    ) -> RetrievalResult:
        rewritten = await self._rewriter.rewrite(query)
        search_text = " ".join([rewritten.normalized, *rewritten.synonyms]).strip()
        if category_prefix:
            safe = category_prefix.replace("\\", "\\\\").replace('"', '\\"').replace("%", "\\%")
            flt = f'category like "{safe}%"'
        else:
            flt = None

        if strategy == "dense":
            vector = await self._embedder.embed_one(search_text)
            raw = self._milvus.search(vector, top_k=self._settings.retrieval_top_k, filter=flt)
        elif strategy == "bm25":
            raw = self._milvus.search_text(search_text, top_k=self._settings.retrieval_top_k, filter=flt)
        else:  # hybrid / hybrid_rerank
            vector = await self._embedder.embed_one(search_text)
            # 融合池要喂得饱精排：hybrid_rerank 至少取 rerank_top_n 条候选
            fusion_top_k = max(
                self._settings.retrieval_top_k,
                self._settings.rerank_top_n if strategy == "hybrid_rerank" and self._reranker else 0,
            )
            raw = self._milvus.hybrid_search(
                query_vector=vector,
                query_text=search_text,
                top_k=fusion_top_k,
                filter=flt,
                rrf_k=self._settings.rrf_k,
                candidates=self._settings.hybrid_candidates,
            )

        evidences = [
            Evidence(
                chunk_id=hit["id"],
                section_path=hit["entity"].get("category", ""),
                question=hit["entity"].get("questions", ""),
                answer=hit["entity"].get("answer", ""),
                score=hit["distance"],
            )
            for hit in raw
        ]

        reranked = False
        if strategy == "hybrid_rerank" and self._reranker is not None and evidences:
            # Reranker.rerank 是 async（httpx.AsyncClient），直接 await
            evidences, reranked = await self._reranker.rerank(
                query, evidences, self._settings.rerank_top_n
            )

        # 置信度语义按策略区分：仅「精排成功后的 relevance_score」与「dense COSINE」
        # 是 0-1 可比阈值；降级时 score 仍是 RRF/BM25 分数，与阈值不可比，不判低置信。
        if strategy == "hybrid_rerank" and reranked:
            low_confidence = not evidences or evidences[0].score < self._settings.retrieval_low_conf_threshold
        elif strategy == "dense":
            low_confidence = not evidences or evidences[0].score < self._settings.retrieval_low_conf_threshold
        else:
            low_confidence = not evidences
        return RetrievalResult(evidences=evidences, low_confidence=low_confidence)

    async def retrieve_multi(
        self,
        queries: list[str],
        strategy: str = "hybrid_rerank",
        category_prefix: str | None = None,
    ) -> RetrievalResult:
        """ch06 扩写多路检索：每条查询独立 retrieve，按 chunk_id 去重取最高分。

        低置信语义放宽：任意一路拿到置信证据即放行（扩写本意是扩召回），
        全部路段都弱才算弱。空查询列表 → 空结果 + 弱置信（调用方走兜底）。
        """
        cleaned = [q.strip() for q in queries if q and q.strip()]
        if not cleaned:
            return RetrievalResult(evidences=[], low_confidence=True)

        best: dict[int, Evidence] = {}
        all_low = True
        for q in cleaned:
            result = await self.retrieve(q, strategy=strategy, category_prefix=category_prefix)
            if not result.low_confidence:
                all_low = False
            for ev in result.evidences:
                cur = best.get(ev.chunk_id)
                if cur is None or ev.score > cur.score:
                    best[ev.chunk_id] = ev

        evidences = sorted(best.values(), key=lambda e: e.score, reverse=True)
        return RetrievalResult(evidences=evidences, low_confidence=all_low)
