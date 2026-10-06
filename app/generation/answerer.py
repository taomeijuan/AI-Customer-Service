import logging
from dataclasses import dataclass, field
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.extraction.service import build_structured_model
from app.generation.prompts import GENERATION_SYSTEM_PROMPT
from app.knowledge.reranker import Evidence

logger = logging.getLogger(__name__)


class AnswerSchema(BaseModel):
    """生成输出：自评 + 带角标回答。"""

    useful: bool = Field(description="提供的知识条目是否足以回答用户问题")
    reason: str = Field(description="useful 判定理由；false 时写清缺什么")
    answer: str = Field(
        description="useful=true 时：带 [n] 来源角标的回答；false 时：简短拒答话术"
    )


@dataclass
class AnswerOutcome:
    useful: bool
    answer: str
    reason: str = ""
    citations: list[dict] = field(default_factory=list)
    low_confidence: bool = False  # useful=false → 生成自评不足，入低置信度池的信号


def _ordered(evidences: list[Evidence]) -> list[Evidence]:
    """首尾放置：最相关放首位、次相关垫底，中间保持原序。"""
    if len(evidences) <= 2:
        return list(evidences)
    return [evidences[0], *evidences[2:], evidences[1]]


def assemble_blocks(evidences: list[Evidence]) -> list[str]:
    """证据编号 [1..K] 的知识文本块。"""
    return [
        f"[{n}] （来源：{ev.section_path}）{ev.answer}"
        for n, ev in enumerate(_ordered(evidences), start=1)
    ]


class Answerer:
    """生成回答：证据组装 → 自评 → 带角标回答或拒答。

    structured_model: llm.with_structured_output(AnswerSchema, include_raw=True,
    method="function_calling")——用 build_structured_model(llm, AnswerSchema) 构建。
    """

    def __init__(self, structured_model: Any) -> None:
        self._structured = structured_model

    async def answer(self, query: str, evidences: list[Evidence]) -> AnswerOutcome:
        blocks = assemble_blocks(evidences)
        knowledge = "\n\n".join(blocks) if blocks else "（无可用知识条目）"
        messages = [
            SystemMessage(GENERATION_SYSTEM_PROMPT),
            HumanMessage(f"知识条目：\n{knowledge}\n\n用户问题：{query}"),
        ]
        result = await self._structured.ainvoke(messages)
        if result["parsing_error"] or result["parsed"] is None:
            logger.warning("answer 解析失败: %s", result["parsing_error"])
            return AnswerOutcome(
                useful=False,
                answer="这个问题我这边暂时没有足够的资料，帮您转人工确认",
                reason=f"生成解析失败: {result['parsing_error']}",
                low_confidence=True,
            )
        parsed = result["parsed"]
        citations = self._build_citations(evidences, parsed.answer) if parsed.useful else []
        return AnswerOutcome(
            useful=parsed.useful,
            answer=parsed.answer,
            reason=parsed.reason,
            citations=citations,
            low_confidence=not parsed.useful,
        )

    @staticmethod
    def _build_citations(evidences: list[Evidence], answer: str) -> list[dict]:
        """角标 [n] 按组装序号映射回原始 chunk；只携带被引用的条目。"""
        cited = []
        for n, ev in enumerate(_ordered(evidences), start=1):
            if f"[{n}]" in answer:
                cited.append(
                    {
                        "n": n,
                        "chunk_id": ev.chunk_id,
                        "section_path": ev.section_path,
                        "question": ev.question,
                        "answer": ev.answer,
                    }
                )
        return cited
