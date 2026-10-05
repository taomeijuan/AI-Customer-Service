from typing import Any

from langchain.tools import tool


def build_query_faq_tool(embedder: Any, milvus_store: Any, settings: Any):
    """query_faq：向量语义检索内核（ch03）。

    入参出参契约与 ch02 关键词查表版完全一致——入参 keyword 即用户问题原文，
    出参 {count, items:[{question, answer, category}]}。替换实现不动前端与编排器。
    """

    @tool
    async def query_faq(keyword: str) -> dict:
        """语义检索常见问题知识库。凡涉及平台政策、费用、流程（退货/退款/邮费/运费/发货/发票等）的问题都必须先调用本工具，禁止凭记忆回答政策。"""
        vector = await embedder.embed_one(keyword)
        hits = milvus_store.search(
            vector,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
        )
        items = [
            {
                "question": h["entity"]["questions"],
                "answer": h["entity"]["answer"],
                "category": h["entity"]["category"],
                "score": round(h["distance"], 4),
            }
            for h in hits
        ]
        return {"count": len(items), "items": items}

    return query_faq
