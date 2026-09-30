from typing import Any

from langchain.messages import HumanMessage

from app.extraction.schemas import AfterSalesExtraction

EXTRACTION_PROMPT = (
    "你是售后工单解析器。从用户描述中提取订单号、诉求类型、期望方案。"
    "订单号未出现则为 null；诉求类型只能是：退款/退货/换货/维修/物流投诉/其他。"
)


def build_structured_model(model: Any) -> Any:
    """method=function_calling：DeepSeek 等兼容上游不支持 json_schema response_format，
    官方支持 function calling（Context7 双向核对 2026-09-30，用户拍板）。"""
    return model.with_structured_output(
        AfterSalesExtraction, include_raw=True, method="function_calling"
    )


class ExtractionService:
    """structured_model: ChatOpenAI.with_structured_output(AfterSalesExtraction, include_raw=True)"""

    def __init__(self, structured_model: Any) -> None:
        self._model = structured_model

    async def extract(self, text: str) -> AfterSalesExtraction:
        result = await self._model.ainvoke(
            [HumanMessage(f"{EXTRACTION_PROMPT}\n\n用户描述：{text}")]
        )
        if result["parsing_error"] or result["parsed"] is None:
            raise ValueError(f"结构化解析失败: {result['parsing_error']}")
        return result["parsed"]
