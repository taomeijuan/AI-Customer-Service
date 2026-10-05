import asyncio
import logging

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


class Embedder:
    """Ollama OpenAI 兼容 /v1/embeddings 客户端（bge-m3，1024 维），支持批量、带一次重试。"""

    def __init__(self, client: httpx.AsyncClient, model: str, retries: int = 1) -> None:
        self._client = client
        self.model = model
        self.retries = retries

    async def embed(self, texts: list[str]) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = await self._client.post(
                    "/embeddings", json={"model": self.model, "input": texts}
                )
                resp.raise_for_status()
                data = sorted(resp.json()["data"], key=lambda d: d["index"])
                if len(data) != len(texts):  # 上游静默截断会破坏 zip 对齐，显式炸掉
                    raise ValueError(f"embeddings 数量不符: {len(data)} != {len(texts)}")
                return [d["embedding"] for d in data]
            except Exception as e:
                last_err = e
                logger.warning("embed attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(0.3 * (attempt + 1))
        raise last_err  # type: ignore[misc]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


def build_embedder(settings: Settings) -> Embedder:
    client = httpx.AsyncClient(base_url=settings.ollama_embed_base_url, timeout=30)
    return Embedder(client=client, model=settings.embed_model)
