import json

import httpx

from app.knowledge.query_rewriter import QueryRewriter, RewriteSchema


def make_rewriter(content: str | None = None, status: int = 200):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read()
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": content
                            or json.dumps({"normalized": "运费规则", "synonyms": ["邮费", "运费"]})
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1")
    return QueryRewriter(client=client, model="test-model"), captured


async def test_rewrite_returns_normalized_and_synonyms():
    r, _ = make_rewriter()
    out = await r.rewrite("邮费咋算啊")
    assert isinstance(out, RewriteSchema)
    assert out.normalized == "运费规则"
    assert "邮费" in out.synonyms


async def test_rewrite_wraps_bare_list():
    """LLM 有时省略外层键：裸数组也要兜住。"""
    r, _ = make_rewriter(content=json.dumps({"normalized": "退货政策", "synonyms": []}))
    out = await r.rewrite("退货咋弄")
    assert out.normalized == "退货政策"


async def test_rewrite_failure_falls_back_to_raw():
    r, _ = make_rewriter(status=500)
    out = await r.rewrite("邮费")
    assert out.normalized == "邮费" and out.synonyms == []  # 降级原话，不炸检索链
