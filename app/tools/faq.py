import asyncio
from typing import Any

from langchain.tools import tool


def build_query_faq_tool(
    session: Any,
    retriever: Any,
    answerer: Any,
    settings: Any,
    conversation_id: int,
):
    """query_faq：混合检索 + 精排 + 生成质控（ch04）。

    入参出参契约与 ch02/ch03 一致（keyword 入，知识结果出）；但内部多了三层质控：
    检索低置信 → 拒答并写池（retrieval_low_conf）；生成自评不足 → 拒答并写池（self_check）；
    正常 → 返回带 [n] 角标的最终回答与 citations（由编排器透传到 done 帧）。
    """

    @tool
    async def query_faq(keyword: str) -> dict:
        """语义检索常见问题知识库。凡涉及平台政策、费用、流程（退货/退款/邮费/运费/发货/发票等）的问题都必须先调用本工具，禁止凭记忆回答政策。"""
        from app.repositories.low_confidence import LowConfidenceRepo

        result = await retriever.retrieve(keyword)
        if result.low_confidence:
            await LowConfidenceRepo(session).record(
                raw_question=keyword,
                source="retrieval_low_conf",
                reason="检索证据不足：融合分数低于阈值或零召回",
                conversation_id=conversation_id,
            )
            return {
                "refused": True,
                "final_answer": "这个问题我这边暂时没有足够的资料，帮您转人工确认",
                "citations": [],
            }

        outcome = await answerer.answer(keyword, result.evidences)
        if not outcome.useful:
            await LowConfidenceRepo(session).record(
                raw_question=keyword,
                source="self_check",
                reason=outcome.reason,
                conversation_id=conversation_id,
            )
            return {"refused": True, "final_answer": outcome.answer, "citations": []}

        return {
            "refused": False,
            "final_answer": outcome.answer,
            "citations": outcome.citations,
        }

    return query_faq
