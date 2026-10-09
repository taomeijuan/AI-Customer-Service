"""ch06 分流器标注样例评测（真实 LLM，验收 1/2/扩写质量）。

按工作要求：Prompt/数据类任务以标注样例跑评替代 TDD。
- intent_eval.jsonl：多轮对话逐轮断言 指代消解关键词 + 意图分类
- expansion_eval.jsonl：退款类问题扩写条数与可用性断言
"""

import json
from pathlib import Path

import pytest
from langchain.messages import AIMessage, HumanMessage

from app.core.llm import get_chat_model
from app.extraction.service import build_structured_model
from app.workflow.expander import ExpansionSchema, Expander
from app.workflow.intent import LangChainIntentClassifier
from app.workflow.resolver import ResolutionSchema, Resolver

DATA = Path(__file__).parent / "data"
STALE_PRONOUNS = ("它能", "它怎么", "这个能", "那个能")  # 消解后的查询不该再裸带指代+动词

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not Path(".env").exists(), reason="需要 .env 真实上游"),
]


def _load(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("case", _load("intent_eval.jsonl"), ids=lambda c: c["id"])
async def test_multiturn_intent_and_resolution(case):
    """验收1+2：多轮切换每轮意图判对、指代补全对；JSON 稳定可解析。"""
    resolver = Resolver(build_structured_model(get_chat_model(), ResolutionSchema))
    classifier = LangChainIntentClassifier(get_chat_model())

    history: list = []
    for ti, turn in enumerate(case["turns"]):
        user = turn["user"]
        resolved = await resolver.resolve(user, history)
        for kw in turn.get("expect_resolved_has") or []:
            assert kw in resolved, f"[{case['id']}#t{ti}] 消解结果缺 {kw!r}：{resolved!r}"
        # 指代+动词的裸指代不应残留（消解失败信号）；原句本就无指代时跳过
        for p in STALE_PRONOUNS:
            if p not in user:
                assert p not in resolved, f"[{case['id']}#t{ti}] 消解残留裸指代：{resolved!r}"

        outcome = await classifier.classify_detail(resolved)
        assert outcome.intent == turn["expected_intent"], (
            f"[{case['id']}#t{ti}] 意图期望 {turn['expected_intent']}，实际 {outcome.intent}"
            f"（confidence={outcome.confidence:.2f}，resolved={resolved!r}）"
        )
        assert outcome.confidence >= 0.5, f"[{case['id']}#t{ti}] 置信度异常低=兜底解析失败信号"
        # 历史滚动：assistant 回复固定用样例提供（不依赖生成质量）
        history.append(HumanMessage(user))
        history.append(AIMessage(turn["assistant"]))


@pytest.mark.parametrize("sample", _load("expansion_eval.jsonl"), ids=lambda s: s["query"][:12])
async def test_expansion_samples(sample):
    """扩写条数落在 2-4，且各条独立可用（无裸指代、非空、去重）。"""
    expander = Expander(build_structured_model(get_chat_model(), ExpansionSchema))
    queries = await expander.expand(sample["query"], context=sample.get("context"))
    lo, hi = sample["expect_count_min"], sample["expect_count_max"]
    assert lo <= len(queries) <= hi, f"扩写数量 {len(queries)} 不在 [{lo},{hi}]：{queries}"
    assert len(set(queries)) == len(queries), f"扩写重复：{queries}"
    for q in queries:
        assert q.strip(), f"扩写含空项：{queries}"
        assert not q.startswith(("它", "这个", "那个")), f"扩写残留裸指代：{q!r}"
