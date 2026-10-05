from sqlalchemy import text

from app.core.config import Settings
from app.db.engine import build_engine


async def test_engine_connects():
    s = Settings(_env_file=None, mysql_database="ecom_cs_test")
    engine, session_factory = build_engine(s)
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT 1"))).scalar() == 1
    await engine.dispose()
