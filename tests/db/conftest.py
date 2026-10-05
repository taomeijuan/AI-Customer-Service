import pytest
from sqlalchemy import text

from app.core.config import Settings
from app.db.engine import build_engine


@pytest.fixture
async def db_session():
    # 每个测试独立 engine：pytest-asyncio 每测试新建事件循环，
    # 跨循环复用连接池会报 "Future attached to a different loop"
    engine, session_factory = build_engine(Settings(_env_file=None, mysql_database="ecom_cs_test"))
    try:
        async with session_factory() as session:
            await session.execute(text("SET FOREIGN_KEY_CHECKS=0"))
            for t in ("messages", "tickets", "conversations", "faq"):
                await session.execute(text(f"TRUNCATE TABLE {t}"))
            await session.execute(text("SET FOREIGN_KEY_CHECKS=1"))
            await session.commit()
            yield session
    finally:
        await engine.dispose()
