import json
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.core.llm import get_chat_model
from app.knowledge.embedder import build_embedder
from app.knowledge.milvus_store import MilvusStore
from app.knowledge.query_rewriter import LangChainRewriter
from app.knowledge.retriever import HybridRetriever
from app.knowledge.reranker import build_reranker

SAMPLES = Path(__file__).parent / "data" / "retrieval_samples.jsonl"


def load_samples():
    return [json.loads(line) for line in SAMPLES.read_text().splitlines() if line.strip()]


pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not (Path(".env").exists() and Path("docs/knowledge").exists()),
        reason="需要 .env 与已建知识库",
    ),
]


@pytest.mark.parametrize("sample", load_samples(), ids=lambda s: s["query"])
async def test_semantic_retrieval_sample(sample):
    """验收1：换说法的问题也能语义召回对应知识（纯检索段，不走生成）。"""
    s = get_settings()
    retriever = HybridRetriever(
        milvus=MilvusStore(uri=s.milvus_uri, collection=s.milvus_collection),
        embedder=build_embedder(s),
        rewriter=LangChainRewriter(get_chat_model()),
        reranker=build_reranker(s),
        settings=s,
    )
    result = await retriever.retrieve(sample["query"], strategy="hybrid_rerank")
    assert result.evidences, f"「{sample['query']}」零召回"
    text = " ".join(e.answer for e in result.evidences)
    assert any(kw in text for kw in sample["expect_any"]), (
        f"「{sample['query']}」召回内容未命中期望关键词: {text[:80]}"
    )
