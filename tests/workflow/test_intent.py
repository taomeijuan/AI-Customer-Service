import json

import httpx
import pytest

from app.workflow.intent import INTENTS, IntentClassifier


def make_classifier(content=None, status=200):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read()
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content or json.dumps({"intent": "订单"})}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1")
    return IntentClassifier(client=client, model="test"), captured


def test_intents_registry():
    assert INTENTS == {"物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"}


async def test_parse_valid_intent():
    c, _ = make_classifier(json.dumps({"intent": "物流"}))
    assert await c.classify("订单1001的物流到哪了") == "物流"


async def test_parse_failure_falls_back_to_business():
    """JSON 解析失败 → 兜底「订单」（业务数据类直进 Agent，最通用）。"""
    c, _ = make_classifier(content="我不是JSON")
    assert await c.classify("随便说点啥") == "订单"


async def test_illegal_value_falls_back():
    c, _ = make_classifier(json.dumps({"intent": "跳舞"}))
    assert await c.classify("跳个舞") == "订单"


async def test_http_error_falls_back():
    c, _ = make_classifier(status=500)
    assert await c.classify("订单1001") == "订单"
