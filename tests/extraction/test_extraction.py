import pytest
from pydantic import ValidationError

from app.extraction.schemas import AfterSalesExtraction
from app.extraction.service import ExtractionService


def test_schema_rejects_bad_issue_type():
    with pytest.raises(ValidationError):
        AfterSalesExtraction(order_no="123", issue_type="白嫖", expected_resolution="x")


def test_schema_order_no_optional():
    m = AfterSalesExtraction(issue_type="其他", expected_resolution="x")
    assert m.order_no is None


class FakeStructured:
    def __init__(self, result):
        self.result = result

    async def ainvoke(self, messages):
        return self.result


async def test_extract_ok():
    parsed = AfterSalesExtraction(
        order_no="A123", issue_type="退款", expected_resolution="全额退款"
    )
    svc = ExtractionService(
        FakeStructured({"raw": None, "parsed": parsed, "parsing_error": None})
    )
    got = await svc.extract("订单A123要退款")
    assert got == parsed


async def test_extract_parse_error_raises():
    svc = ExtractionService(
        FakeStructured({"raw": "xx", "parsed": None, "parsing_error": "boom"})
    )
    with pytest.raises(ValueError):
        await svc.extract("乱七八糟")
