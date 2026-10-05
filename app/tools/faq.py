from typing import Any

from langchain.tools import tool

from app.repositories.faq import FaqRepo


def build_query_faq_tool(session: Any):
    """FAQ 查询工具。会话级工厂：注入请求级 AsyncSession（conversation 无关，可直接用）。"""

    @tool
    async def query_faq(keyword: str) -> dict:
        """根据关键词检索常见问题知识库。凡涉及平台政策、费用、流程（退货/退款/邮费/运费/发货/发票等）的问题都必须先调用本工具，禁止凭记忆回答政策。"""
        rows = await FaqRepo(session).search(keyword)
        return {"count": len(rows), "items": rows}

    return query_faq
