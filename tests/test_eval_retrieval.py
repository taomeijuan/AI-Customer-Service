import json
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.knowledge.embedder import build_embedder
from app.knowledge.milvus_store import MilvusStore
from app.tools.faq import build_query_faq_tool

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
    """验收1：换说法的问题也能语义召回对应知识。"""
    s = get_settings()
    faq = build_query_faq_tool(build_embedder(s), MilvusStore(uri=s.milvus_uri, collection=s.milvus_collection), s)
    out = await faq.ainvoke({"keyword": sample["query"]})
    assert out["count"] > 0, f"「{sample['query']}」零召回"
    text = " ".join(item["answer"] for item in out["items"])
    assert any(kw in text for kw in sample["expect_any"]), (
        f"「{sample['query']}」召回内容未命中期望关键词: {text[:80]}"
    )
