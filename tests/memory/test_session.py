import asyncio

import pytest
from langchain.messages import AIMessage, HumanMessage

from app.memory.session import SessionStore


async def test_create_returns_new_id_and_empty_history():
    store = SessionStore()
    cid = await store.create()
    assert await store.get(cid) == []


async def test_get_missing_returns_none():
    store = SessionStore()
    assert await store.get("nope") is None


async def test_append_then_get_roundtrip():
    store = SessionStore()
    cid = await store.create()
    msgs = [HumanMessage("你好"), AIMessage("您好")]
    await store.append(cid, msgs)
    assert await store.get(cid) == msgs


async def test_append_missing_raises():
    store = SessionStore()
    with pytest.raises(KeyError):
        await store.append("nope", [HumanMessage("x")])


async def test_concurrent_append_safe():
    store = SessionStore()
    cid = await store.create()
    await asyncio.gather(
        *[store.append(cid, [HumanMessage(f"m{i}")]) for i in range(50)]
    )
    assert len(await store.get(cid)) == 50
