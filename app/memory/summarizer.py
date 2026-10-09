"""ch07 T7：后台异步摘要任务——触发看用量、一段一行只追加、压完不回头。

设计要点（需求 2）：
- **不阻塞用户轮**：调用方 asyncio.create_task 起任务即返回，本模块自身全程 await
  独立 session（不共享用户轮的会话对象）。
- **只追加不回炉**：新段只压「本批层2溢出轮」，旧梗概仅作背景给模型看、不参与
  合并——防同一事实被反复有损压缩（用户 DDL 注释原义）。
- **单飞**：同会话同时只有一个任务；重复触发 log skip。
- 幂等：段+边界+投影在仓储单事务里推进；任务失败边界不动，下轮自然重试。
"""

import asyncio
import logging
from typing import Any

from langchain_core.messages import BaseMessage

from app.memory.tokens import estimate_tokens

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """你是客服会话压缩器。把下面这批对话压成一段 40-160 字的梗概，只提炼事实与诉求：
- 问过/看过的商品（名称）
- 用户报过的订单号、手机号等关键信息（数字原样保留）
- 明确表达的诉求（退款/换货/查物流/投诉…）
- 尚未解决、后续轮次要接着办的事

规则：
1. 对话里没出现的内容一个字都不许编
2. 寒暄、客套、闲聊一律不留
3. 不加评价、不写建议，直接输出梗概正文（不要任何前缀）"""

MAX_SUMMARY_CHARS = 220


class Summarizer:
    def __init__(self, model: Any, session_factory: Any, settings: Any) -> None:
        self._model = model
        self._sf = session_factory
        self._settings = settings
        self._inflight: set[int] = set()  # per-cid 单飞

    def schedule(
        self, conversation_id: int, batch: list[tuple[int, BaseMessage]], projection: str | None
    ) -> asyncio.Task | None:
        """有在途任务即 skip（留痕）；否则起后台任务，立即返回不阻塞。"""
        if not batch:
            return None
        if conversation_id in self._inflight:
            logger.info("summary skip: conv=%s 已有在途任务", conversation_id)
            return None
        self._inflight.add(conversation_id)
        task = asyncio.create_task(self._run(conversation_id, batch, projection))
        task.add_done_callback(lambda _t: self._inflight.discard(conversation_id))
        return task

    async def _run(
        self, conversation_id: int, batch: list[tuple[int, BaseMessage]], projection: str | None
    ) -> None:
        from app.repositories.summaries import SummariesRepo

        first_id, last_id = batch[0][0], batch[-1][0]
        est = sum(estimate_tokens(str(m.text or "")) for _, m in batch)
        logger.info(
            "summary start: conv=%s 批=%d条(id %d→%d) ≈%d token",
            conversation_id, len(batch), first_id, last_id, est,
        )
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        try:
            content = await self._summarize(batch, projection)
            async with self._sf() as session:
                seq = await SummariesRepo(session).append_segment(
                    conversation_id,
                    from_msg_id=first_id,
                    upto_msg_id=last_id,
                    content=content,
                    projection_budget=self._settings.summary_inject_reserve,
                )
            logger.info(
                "summary done: conv=%s 第%d段(id≤%d) %d字 耗时%dms",
                conversation_id, seq, last_id, len(content), int((loop.time() - t0) * 1000),
            )
        except Exception as e:
            # 边界未推进，下一轮自然重试；只留痕不炸主流程
            logger.warning("summary fail: conv=%s err=%s", conversation_id, e)

    async def _summarize(self, batch: list[tuple[int, BaseMessage]], projection: str | None) -> str:
        from langchain.messages import HumanMessage, SystemMessage

        lines: list[str] = []
        if projection:
            lines.append(f"（更早对话的背景，不用复述进新梗概）：{projection}")
        for _, m in batch:
            role = {"human": "用户", "ai": "客服", "tool": "工具"}.get(m.type, m.type)
            text = (m.text or "").strip()
            if m.type == "ai" and getattr(m, "tool_calls", None):
                text = "（调用工具）" + " ".join(tc.get("name", "") for tc in m.tool_calls)
            if text:
                lines.append(f"{role}: {text}")
        result = await self._model.ainvoke(
            [SystemMessage(SUMMARY_SYSTEM_PROMPT), HumanMessage("\n".join(lines))]
        )
        content = (result.text or "").strip().replace("\n", " ")
        if len(content) > MAX_SUMMARY_CHARS:
            content = content[:MAX_SUMMARY_CHARS]
        if not content:
            raise ValueError("empty summary")
        return content
