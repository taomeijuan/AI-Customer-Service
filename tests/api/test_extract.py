from fastapi.testclient import TestClient

from app.extraction.schemas import AfterSalesExtraction
from app.extraction.service import ExtractionService
from app.main import create_app
from tests.extraction.test_extraction import FakeStructured


def make_client(structured):
    app = create_app()
    app.state.extract_service = ExtractionService(structured)
    return TestClient(app)


def test_extract_ok():
    parsed = AfterSalesExtraction(
        order_no="A123", issue_type="退款", expected_resolution="退钱"
    )
    client = make_client(
        FakeStructured({"raw": None, "parsed": parsed, "parsing_error": None})
    )
    resp = client.post("/api/extract", json={"text": "订单A123要退款"})
    assert resp.status_code == 200
    assert resp.json() == {
        "order_no": "A123",
        "issue_type": "退款",
        "expected_resolution": "退钱",
    }


def test_extract_parse_error_422():
    client = make_client(
        FakeStructured({"raw": "x", "parsed": None, "parsing_error": "boom"})
    )
    resp = client.post("/api/extract", json={"text": "乱七八糟"})
    assert resp.status_code == 422


def test_extract_missing_field_422():
    client = make_client(create_app())  # 未注入也应有校验层兜底
    resp = client.post("/api/extract", json={})
    assert resp.status_code == 422


def test_extract_empty_text_422():
    client = make_client(create_app())
    resp = client.post("/api/extract", json={"text": ""})
    assert resp.status_code == 422
