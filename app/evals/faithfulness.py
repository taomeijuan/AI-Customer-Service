import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.extraction.service import build_structured_model
from app.repositories.faith_cases import FaithCasesRepo

logger = logging.getLogger(__name__)

JUDGE_PROMPT = (
    "你是忠实度裁判。判断【回答】是否完全被【知识条目】支持：\n"
    "- 回答中任何一条事实性陈述（数字、时间、承诺、规则）在知识条目里找不到依据 → 不支持。\n"
    "- 回答对知识的转述、概括、口语化不算编造。\n"
    "只输出 JSON：{\"faithful\": true/false, \"reason\": \"...\"}"
)


class FaithSchema(BaseModel):
    faithful: bool = Field(description="回答是否完全被知识条目支持")
    reason: str = Field(description="判定理由；不支持时写清编在哪一句")


@dataclass
class FaithVerdict:
    faithful: bool
    reason: str


class FaithfulnessJudge:
    """LLM-as-judge：判编造 → 落 faith_cases 台账（一题一行跨轮追溯）。"""

    def __init__(self, llm: Any, repo: FaithCasesRepo, judge_model: str = "deepseek-chat") -> None:
        self._structured = build_structured_model(llm, FaithSchema)
        self._repo = repo
        self.judge_model = judge_model

    async def verify(
        self,
        eval_id: str,
        bucket: str,
        query: str,
        strategy: str,
        answer: str,
        citations: list[dict],
    ) -> FaithVerdict:
        knowledge = "\n\n".join(
            f"[{c['n']}] {c['question']}：{c['answer']}" for c in citations
        )
        result = await self._structured.ainvoke(
            [HumanMessageIfAvailable(f"{JUDGE_PROMPT}\n\n知识条目：\n{knowledge}\n\n【回答】：{answer}")]
        )
        if result["parsing_error"] or result["parsed"] is None:
            logger.warning("faithfulness judge 解析失败: %s", result["parsing_error"])
            return FaithVerdict(faithful=True, reason="judge 解析失败（保守放行）")

        parsed = result["parsed"]
        if not parsed.faithful:
            await self._repo.record(
                eval_id=eval_id,
                bucket=bucket,
                query=query,
                strategy=strategy,
                answer=answer,
                reason=parsed.reason,
                citations=citations,
                judge_model=self.judge_model,
            )
        return FaithVerdict(faithful=parsed.faithful, reason=parsed.reason)


def HumanMessageIfAvailable(text: str):
    from langchain.messages import HumanMessage

    return HumanMessage(text)


def aggregate_report(rows: list[dict]) -> dict:
    """分桶聚合：每桶各项指标均值 + 题数。"""
    out: dict[str, dict] = {}
    for r in rows:
        bucket = out.setdefault(r["bucket"], {"n": 0})
        bucket["n"] += 1
        for k, v in r.items():
            if k in ("bucket", "n") or not isinstance(v, (int, float)):
                continue
            bucket[k] = bucket.get(k, 0.0) + v
    for bucket in out.values():
        n = bucket.pop("n")
        for k in list(bucket):
            bucket[k] = bucket[k] / n
        bucket["n"] = n
    return out
