import httpx

from app.core.config import Settings


class Embedder:
    """Ollama OpenAI 兼容 /v1/embeddings 客户端（bge-m3，1024 维），支持批量。"""

    def __init__(self, client: httpx.AsyncClient, model: str) -> None:
        self._client = client
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.post(
            "/embeddings", json={"model": self.model, "input": texts}
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


def build_embedder(settings: Settings) -> Embedder:
    client = httpx.AsyncClient(base_url=settings.ollama_embed_base_url, timeout=30)
    return Embedder(client=client, model=settings.embed_model)
