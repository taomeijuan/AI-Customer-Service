import json
from pathlib import Path

import pytest

from app.extraction.schemas import AfterSalesExtraction
from app.extraction.service import ExtractionService

SAMPLES = Path(__file__).parent / "data" / "after_sales_samples.jsonl"


def load_samples():
    return [json.loads(line) for line in SAMPLES.read_text().splitlines() if line.strip()]


pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not Path(".env").exists(), reason="需要 .env 真实上游"),
]


@pytest.mark.parametrize("sample", load_samples(), ids=lambda s: s["text"][:12])
async def test_extraction_sample(sample):
    from app.core.llm import get_chat_model

    svc = ExtractionService(
        get_chat_model().with_structured_output(AfterSalesExtraction, include_raw=True)
    )
    got = await svc.extract(sample["text"])
    exp = sample["expected"]
    assert got.issue_type == exp["issue_type"]
    assert got.order_no == exp["order_no"]
    assert got.expected_resolution.strip()
