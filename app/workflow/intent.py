"""意图识别节点（ch06 四件套正式版）：选择题枚举 + 强制 JSON + few-shot + 其他兜底。

接口兼容：classify(query) 沿用 ch05 签名；ch06 新增 classify_detail 返回
{intent, confidence}，图节点用它做置信度入池（<0.6 → low_confidence_questions）。
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

INTENT_NAMES = ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他"]
IntentName = Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他"]
INTENTS = set(INTENT_NAMES)
FALLBACK_INTENT: IntentName = "订单"  # 兜底=业务数据类：直进 Agent 最通用
INTENT_LOW_CONF_THRESHOLD = 0.6  # 低于此值入低置信池（ch09 数据飞轮入口）

INTENT_PROMPT = (
    "你是电商客服的意图分类器。从下面八个类别里选一个最贴近用户意图的（只选一个）：\n"
    "1. 物流 —— 快递到哪了、多久送到、发货进度\n"
    "2. 订单 —— 查订单状态、订单里有什么、订单金额\n"
    "3. 商品咨询 —— 商品功能、参数、怎么用\n"
    "4. 退款退货 —— 退货退款政策、邮费规则、能不能退\n"
    "5. 售后 —— 换货、维修、发票等售后办理\n"
    "6. 投诉 —— 不满、要投诉、态度差\n"
    "7. 闲聊 —— 打招呼、寒暄、与购物无关\n"
    "8. 其他 —— 拿不准或无法归入以上任何一类。宁可归其他，禁止硬塞进业务类别\n\n"
    "边界样例：\n"
    '- 「它啥时候能到？」→ 物流\n'
    '- 「这个能退吗？」→ 退款退货\n'
    '- 「你们的东西是正品吗？」→ 商品咨询\n'
    '- 「我要投诉你们客服！」→ 投诉\n'
    '- 「在吗？」→ 闲聊\n'
    '- 「帮我看看那个单子」→ 其他（看不出具体要办什么）\n'
    '- 「发票怎么开」→ 售后\n\n'
    '只输出 JSON：{"intent": "<类别名>", "confidence": <0到1的小数>}\n'
    "不要解释。"
)


class IntentSchema(BaseModel):
    intent: IntentName = Field(description="八个类别之一")
    confidence: float = Field(default=0.5, ge=0, le=1, description="判断置信度 0-1")


@dataclass
class ClassifyOutcome:
    intent: IntentName
    confidence: float


class IntentClassifier:
    """httpx 直连形态（测试用 MockTransport 注入）。失败一律兜底「订单」。"""

    def __init__(self, client: httpx.AsyncClient, model: str, retries: int = 1) -> None:
        self._client = client
        self.model = model
        self.retries = retries

    async def _request(self, query: str) -> IntentSchema:
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
        schema = IntentSchema.model_validate_json(content)
        if schema.intent not in INTENTS:
            raise ValueError(f"非法意图: {schema.intent}")
        return schema

    async def classify_detail(self, query: str) -> ClassifyOutcome:
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                schema = await self._request(query)
                return ClassifyOutcome(intent=schema.intent, confidence=schema.confidence)
            except Exception as e:
                last_err = e
                logger.warning("intent classify attempt %d failed: %s", attempt + 1, e)
        logger.warning("intent fallback to %s: %s", FALLBACK_INTENT, last_err)
        return ClassifyOutcome(intent=FALLBACK_INTENT, confidence=0.0)

    async def classify(self, query: str) -> IntentName:
        return (await self.classify_detail(query)).intent


class LangChainIntentClassifier:
    """生产形态：LangChain 结构化输出（function_calling），失败兜底。"""

    def __init__(self, llm: Any) -> None:
        from app.extraction.service import build_structured_model

        self._structured = build_structured_model(llm, IntentSchema)

    async def classify_detail(self, query: str) -> ClassifyOutcome:
        from langchain.messages import HumanMessage

        try:
            result = await self._structured.ainvoke(
                [HumanMessage(f"{INTENT_PROMPT}\n\n用户消息：{query}")]
            )
            if result["parsing_error"] or result["parsed"] is None:
                raise ValueError(str(result["parsing_error"]))
            parsed = result["parsed"]
            intent = parsed.intent if parsed.intent in INTENTS else FALLBACK_INTENT
            return ClassifyOutcome(intent=intent, confidence=parsed.confidence)
        except Exception as e:
            logger.warning("intent fallback to %s: %s", FALLBACK_INTENT, e)
            return ClassifyOutcome(intent=FALLBACK_INTENT, confidence=0.0)

    async def classify(self, query: str) -> IntentName:
        return (await self.classify_detail(query)).intent
