import httpx

from app.knowledge.reranker import Reranker, build_reranker
from app.knowledge.reranker import Evidence


def _chunk(cid: int) -> Evidence:
    return Evidence(chunk_id=cid, section_path=f"售后政策/节{cid}", question=f"问题{cid}", answer=f"答案正文{cid}")


def make_reranker(results=None, status: int = 200):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read()
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(
            200,
            json={
                "results": results
                or [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1")
    return Reranker(client=client, model="BAAI/bge-reranker-v2-m3"), captured


async def test_rerank_orders_by_relevance():
    r, captured = make_reranker()
    docs = [_chunk(1), _chunk(2), _chunk(3)]
    out, reranked = await r.rerank("query", docs, top_n=2)
    assert reranked is True
    assert [e.chunk_id for e in out] == [3, 1]  # 按 results.index 映射回原文档
    assert out[0].score == 0.9
    assert b"AF" not in captured["body"]
    assert b"answer" in captured["body"] or b"\xe7\xad\x94\xe6\xa1\x88" in captured["body"]  # 文档文本入请求


async def test_rerank_api_error_flags_degraded():
    """评审 M3：降级必须显式标记——原序返回时 score 仍是 RRF 分数，不可与阈值比较。"""
    r, _ = make_reranker(status=500)
    docs = [_chunk(1), _chunk(2)]
    out, reranked = await r.rerank("query", docs, top_n=2)
    assert [e.chunk_id for e in out] == [1, 2]
    assert reranked is False


def test_build_reranker_without_key_returns_none():
    from types import SimpleNamespace

    s = SimpleNamespace(rerank_api_base="https://x/v1", rerank_api_key="", rerank_model="m")
    assert build_reranker(s) is None
