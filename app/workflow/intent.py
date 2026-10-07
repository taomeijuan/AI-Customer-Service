"""意图识别节点（ch05 最简版）：LLM 七类 JSON 分类 + 失败兜底。"""

import json
import logging
from typing import Literal

from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

INTENTS = {"物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"}
IntentName = Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"]
FALLBACK_INTENT: IntentName = "订单"  # 兜底=业务数据类：直进 Agent 最通用

INTENT_PROMPT = (
    "你是电商客服的意图分类器。把用户消息判成以下七类之一，只输出 JSON：\n"
    '{"intent": "物流|订单|商品咨询|退款退货|售后|投诉|闲聊"}\n\n'
    "判例：\n"
    '- 问快递/到哪了/多久送到 → "物流"\n'
    '- 查订单状态/订单里有什么/订单号相关 → "订单"\n'
    '- 问商品功能/怎么用/参数 → "商品咨询"\n'
    '- 退货退款政策/邮费规则 → "退款退货"\n'
    '- 换货/维修/发票等售后办理 → "售后"\n'
    '- 不满/投诉/态度差 → "投诉"\n'
    '- 打招呼/寒暄/与购物无关 → "闲聊"\n\n'
    "只输出 JSON，不要解释。"
)


class IntentSchema(BaseModel):
    intent: IntentName = Field(description="七类意图之一")


class IntentClassifier:
    """httpx 直连形态（测试用 MockTransport 注入）。失败一律兜底「订单」。"""

    def __init__(self, client: httpx.AsyncClient, model: str, retries: int = 1) -> None:
        self._client = client
        self.model = model
        self.retries = retries

    async def classify(self, query: str) -> IntentName:
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = await self._client.post(
                    "/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": f"{INTENT_PROMPT}\n\n用户消息：{query}"}
                        ],
                        "temperature": 0,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                intent = IntentSchema.model_validate_json(content).intent
                if intent not in INTENTS:
                    raise ValueError(f"非法意图: {intent}")
                return intent
            except Exception as e:
                last_err = e
                logger.warning("intent classify attempt %d failed: %s", attempt + 1, e)
        logger.warning("intent fallback to %s: %s", FALLBACK_INTENT, last_err)
        return FALLBACK_INTENT


class LangChainIntentClassifier:
    """生产形态：LangChain 结构化输出（function_calling），失败兜底。"""

    def __init__(self, llm: Any) -> None:
        from app.extraction.service import build_structured_model

        self._structured = build_structured_model(llm, IntentSchema)

    async def classify(self, query: str) -> IntentName:
        from langchain.messages import HumanMessage

        try:
            result = await self._structured.ainvoke(
                [HumanMessage(f"{INTENT_PROMPT}\n\n用户消息：{query}")]
            )
            if result["parsing_error"] or result["parsed"] is None:
                raise ValueError(str(result["parsing_error"]))
            intent = result["parsed"].intent
            return intent if intent in INTENTS else FALLBACK_INTENT
        except Exception as e:
            logger.warning("intent fallback to %s: %s", FALLBACK_INTENT, e)
            return FALLBACK_INTENT
