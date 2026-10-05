import httpx

from app.knowledge.embedder import Embedder, build_embedder
from app.core.config import Settings


def make_embedder() -> Embedder:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.1, 0.2], "index": 0},
                    {"embedding": [0.3, 0.4], "index": 1},
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1")
    return Embedder(client=client, model="bge-m3"), captured


async def test_batch_embed_preserves_order():
    emb, captured = make_embedder()
    out = await emb.embed(["问题一", "问题二"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]
    assert b"bge-m3" in captured["body"]
    assert b"\xe9\x97\xae\xe9\xa2\x98\xe4\xb8\x80" in captured["body"]  # 问题一


async def test_embed_one():
    emb, _ = make_embedder()
    assert await emb.embed_one("邮费") == [0.1, 0.2]


async def test_build_embedder_from_settings():
    s = Settings(_env_file=None)
    e = build_embedder(s)
    assert e.model == s.embed_model
