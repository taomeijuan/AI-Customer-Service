import pytest

from app.generation.answerer import Answerer, assemble_blocks
from app.generation.prompts import GENERATION_SYSTEM_PROMPT
from app.knowledge.reranker import Evidence


def make_ev(rank: int) -> Evidence:
    return Evidence(chunk_id=100 + rank, section_path=f"售后政策/节{rank}", question=f"问题{rank}", answer=f"知识内容{rank}", score=1.0 - rank / 10)


def test_first_and_last_placement():
    """最相关放首尾，中间按序：利用 LLM 首尾注意力。"""
    evs = [make_ev(1), make_ev(2), make_ev(3), make_ev(4), make_ev(5)]  # rank1 最相关
    blocks = assemble_blocks(evs)
    assert blocks[0].startswith("[1]") and "知识内容1" in blocks[0]
    assert blocks[-1].startswith("[2]") and "知识内容2" in blocks[-1]  # 次相关垫底
    assert [b.split("]")[0] + "]" for b in blocks] == ["[1]", "[3]", "[4]", "[5]", "[2]"]


class FakeStructured:
    def __init__(self, parsed):
        self.parsed = parsed

    async def ainvoke(self, messages):
        return {"raw": None, "parsed": self.parsed, "parsing_error": None}


class Out:
    def __init__(self, useful, reason, answer):
        self.useful = useful
        self.reason = reason
        self.answer = answer


async def test_useful_true_returns_cited_answer():
    llm_out = Out(True, "知识足够", "退款[1]通常1-3个工作日[2]原路退回")
    a = Answerer(FakeStructured(llm_out))
    outcome = await a.answer("退款多久到账", [make_ev(1), make_ev(2)])
    assert outcome.useful is True
    assert outcome.answer.startswith("退款[1]")
    assert outcome.citations[0]["chunk_id"] == 101  # 角标 [1] ↔ 证据位次映射
    assert outcome.citations[0]["section_path"] == "售后政策/节1"


async def test_useful_false_flags_refusal():
    llm_out = Out(False, "知识库里没有量子速递的内容", "这个问题我这边暂时答不了")
    a = Answerer(FakeStructured(llm_out))
    outcome = await a.answer("量子速递多久到", [make_ev(1)])
    assert outcome.useful is False
    assert outcome.low_confidence is True  # 生成自评不足 → 入池信号


def test_negative_knowledge_in_prompt():
    """System Prompt 列明禁止承诺的负面知识。"""
    for kw in ("不承诺", "到账时间", "优惠", "库存", "赔偿"):
        assert kw in GENERATION_SYSTEM_PROMPT


async def test_parsing_error_treated_as_low_confidence():
    a = Answerer(FakeStructured({"raw": "x", "parsed": None, "parsing_error": "boom"}))
    outcome = await a.answer("问题", [make_ev(1)])
    assert outcome.useful is False and outcome.low_confidence is True
