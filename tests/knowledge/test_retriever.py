import pytest

from app.knowledge.retriever import HybridRetriever
from app.knowledge.reranker import Evidence


class FakeRewriter:
    def __init__(self, normalized="运费规则", synonyms=("邮费", "运费")):
        self.normalized = normalized
        self.synonyms = list(synonyms)
        self.called_with = None

    async def rewrite(self, query):
        self.called_with = query
        from app.knowledge.query_rewriter import RewriteSchema

        return RewriteSchema(normalized=self.normalized, synonyms=self.synonyms)


class FakeEmbedder:
    async def embed_one(self, text):
        return [1.0, 0.0]


class FakeMilvus:
    def __init__(self, hybrid_hits=None, dense_hits=None, text_hits=None):
        self.hybrid_hits = hybrid_hits or []
        self.dense_hits = dense_hits or []
        self.text_hits = text_hits or []
        self.calls: dict = {}

    def _to_hits(self, hits):
        return [
            {
                "id": cid,
                "distance": score,
                "entity": {
                    "questions": f"问题{cid}",
                    "answer": f"答案{cid}",
                    "category": "售后政策",
                    "content_type": "policy",
                },
            }
            for cid, score in hits
        ]

    def search(self, vector, top_k=3, score_threshold=None, filter=None):
        self.calls["search"] = self.calls.get("search", 0) + 1
        return self._to_hits(self.dense_hits)[:top_k]

    def search_text(self, query_text, top_k=3, filter=None):
        self.calls["search_text"] = {"text": query_text}
        return self._to_hits(self.text_hits)[:top_k]

    def hybrid_search(self, query_vector, query_text, top_k=10, filter=None, rrf_k=60, candidates=50):
        self.calls["hybrid_search"] = {"text": query_text, "filter": filter}
        return self._to_hits(self.hybrid_hits)[:top_k]


class FakeReranker:
    """按精排位次给 relevance_score（首位 0.9 递减），模拟真实 bge-reranker 分数形态。"""

    def __init__(self):
        self.called = False

    def rerank(self, query, evidences, top_n):
        self.called = True
        out = []
        for i, e in enumerate(evidences[:top_n]):
            e.score = 0.9 - 0.1 * i
            out.append(e)
        return out


def make(milvus, rewriter=None, reranker=None):
    return HybridRetriever(
        milvus=milvus,
        embedder=FakeEmbedder(),
        rewriter=rewriter or FakeRewriter(),
        reranker=reranker,
        settings=type(
            "S",
            (),
            {
                "retrieval_top_k": 5,
                "hybrid_candidates": 50,
                "rrf_k": 60,
                "rerank_top_n": 3,
                "retrieval_low_conf_threshold": 0.45,
            },
        )(),
    )


async def test_hybrid_rerank_pipeline():
    milvus = FakeMilvus(hybrid_hits=[(1, 0.02), (2, 0.015), (3, 0.01)])
    rewriter = FakeRewriter()
    retriever = make(milvus, rewriter, FakeReranker())
    result = await retriever.retrieve("邮费多少", strategy="hybrid_rerank")
    assert rewriter.called_with == "邮费多少"
    assert milvus.calls["hybrid_search"]["text"] == "运费规则 邮费 运费"  # 改写+同义词进 BM25 路
    assert len(result.evidences) <= 3
    assert all(isinstance(e, Evidence) for e in result.evidences)
    assert result.low_confidence is False


async def test_dense_only_strategy():
    milvus = FakeMilvus(dense_hits=[(1, 0.8)])
    retriever = make(milvus)
    result = await retriever.retrieve("邮费", strategy="dense")
    assert milvus.calls.get("search", 0) == 1
    assert "hybrid_search" not in milvus.calls
    assert result.evidences[0].chunk_id == 1


async def test_bm25_only_strategy_uses_expanded_text():
    milvus = FakeMilvus(text_hits=[(2, 30.0)])
    retriever = make(milvus)
    result = await retriever.retrieve("邮费", strategy="bm25")
    assert milvus.calls["search_text"]["text"] == "运费规则 邮费 运费"
    assert result.evidences[0].chunk_id == 2


async def test_category_prefix_filter_passed_down():
    milvus = FakeMilvus(hybrid_hits=[(1, 0.02)])
    retriever = make(milvus, reranker=FakeReranker())
    await retriever.retrieve("邮费", strategy="hybrid", category_prefix="售后")
    assert 'category like "售后%"' in milvus.calls["hybrid_search"]["filter"]


async def test_low_confidence_flag_when_all_scores_low():
    class LowScoreReranker(FakeReranker):
        def rerank(self, query, evidences, top_n):
            self.called = True
            for e in evidences[:top_n]:
                e.score = 0.001  # 精排分低于阈值
            return evidences[:top_n]

    milvus = FakeMilvus(hybrid_hits=[(1, 0.001)])
    retriever = make(milvus, reranker=LowScoreReranker())
    result = await retriever.retrieve("邮费", strategy="hybrid_rerank")
    assert result.low_confidence is True  # top1 精排分低于阈值 → 检索侧拒答信号
