"""造几段含「邮费/运费」问答的历史会话写入 messages，供挖知识任务有料可挖。

用法: uv run python scripts/seed_dialogs.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.db.engine import build_engine
from app.db.models import Conversation, Message

DIALOGS = [
    [
        ("在你们买东西运费怎么算的呀", "普通订单满99元包邮，不满99元收取8元邮费，偏远地区每单加收10元哦"),
        ("那如果是我自己不想要了呢", "七天无理由退货的话，非质量问题退货运费需要买家承担，8元每单"),
    ],
    [
        ("东西还没发货，多久能发呀", "现货商品48小时内发货，预售的按商品页面标注时间发货"),
        ("发货了怎么查物流", "在「我的订单」点进对应订单就能看实时物流轨迹啦"),
    ],
    [
        ("买的东西有质量问题怎么办", "质量问题30天内可以退货，运费平台全额承担，提供照片凭证就可以"),
        ("退款多久能到账", "审核通过后1-3个工作日原路退回，银行卡可能要3-7个工作日"),
    ],
]


async def main() -> int:
    settings = get_settings()
    engine, session_factory = build_engine(settings)
    try:
        async with session_factory() as session:
            count = 0
            for dialog in DIALOGS:
                conv = Conversation(user_id="seed-miner")
                session.add(conv)
                await session.flush()
                for user, assistant in dialog:
                    session.add(
                        Message(conversation_id=conv.id, role="user", content=user)
                    )
                    session.add(
                        Message(
                            conversation_id=conv.id,
                            role="assistant",
                            content=assistant,
                        )
                    )
                count += 1
            await session.commit()
            print(f"✓ 已写入 {count} 段种子会话（共 {count * 4} 条消息），可运行挖知识：")
            print("  uv run python -m app.jobs.mine_qa")
            return 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
