import json

from fastapi.testclient import TestClient
from langchain.messages import HumanMessage

from app.chains.chat import ChatService
from app.memory.session import SessionStore
from app.main import create_app
from tests.chains.test_chat_service import FakeModel


def make_client(store=None, model=None):
    app = create_app()
    app.state.store = store or SessionStore()
    app.state.chat_service = ChatService(model or FakeModel(["您好", "，", "在的"]))
    return TestClient(app)


def sse_events(resp):
    out = []
    for block in resp.text.split("\n\n"):
        if not block.strip():
            continue
        ev, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1])
        out.append((ev, data))
    return out


def test_sse_event_sequence_and_history_persisted():
    store = SessionStore()
    client = make_client(store=store)
    resp = client.post(
        "/api/chat/stream", json={"conversation_id": None, "message": "你好"}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = sse_events(resp)
    assert events[0][0] == "meta" and events[0][1]["conversation_id"]
    assert [e for e, _ in events if e == "delta"] == ["delta"] * 3
    assert events[-1][0] == "done"
    cid = events[0][1]["conversation_id"]
    history = store._sessions[cid]  # 白盒断言：流结束后回填
    assert [type(m).__name__ for m in history] == ["HumanMessage", "AIMessage"]
    assert history[1].content == "您好，在的"


def test_second_turn_receives_first_turn_history():
    store = SessionStore()
    model = FakeModel(["好的"])
    client = make_client(store=store, model=model)
    r1 = client.post("/api/chat/stream", json={"message": "第一句"})
    cid = sse_events(r1)[0][1]["conversation_id"]
    client.post("/api/chat/stream", json={"conversation_id": cid, "message": "第二句"})
    second_call_messages = model.seen
    assert second_call_messages[1].content == "第一句"  # 上轮 human
    assert second_call_messages[2].type == "ai"  # 上轮 ai 回复
    assert second_call_messages[-1].content == "第二句"


def test_unknown_conversation_id_404():
    client = make_client()
    resp = client.post(
        "/api/chat/stream",
        json={"conversation_id": "ghost", "message": "hi"},
    )
    assert resp.status_code == 404


def test_upstream_error_emits_error_event_and_no_history_append():
    class Boom(FakeModel):
        async def astream(self, messages):
            raise RuntimeError("upstream down")
            yield  # pragma: no cover

    store = SessionStore()
    client = make_client(store=store, model=Boom([]))
    resp = client.post("/api/chat/stream", json={"message": "hi"})
    assert resp.status_code == 200  # SSE 已起流，错误走事件而非状态码
    events = sse_events(resp)
    assert events[-1][0] == "error" and "message" in events[-1][1]
    assert store._sessions == {}  # 出错不回填，且新会话空壳已被清理


def test_empty_message_422():
    client = make_client()
    resp = client.post("/api/chat/stream", json={"message": ""})
    assert resp.status_code == 422


def test_token_budget_covers_current_input():
    """回归：预算必须扣除当前输入，否则用户超长消息可把历史全部挤出预算而不报警。"""
    from types import SimpleNamespace

    from app.memory.trimmer import count_tokens

    huge = "很" * 50
    store = SessionStore()
    model = FakeModel(["回"])
    app = create_app()
    app.state.store = store
    app.state.chat_service = ChatService(model)
    # 预算 = 当前输入 token + 2：扣除当前输入后历史只剩一条 AI 回复的空间
    app.state.settings = SimpleNamespace(
        token_budget=count_tokens(HumanMessage(huge)) + 2
    )
    client = TestClient(app)

    r1 = client.post("/api/chat/stream", json={"message": "第一句"})
    cid = sse_events(r1)[0][1]["conversation_id"]
    client.post("/api/chat/stream", json={"conversation_id": cid, "message": "第二句"})
    client.post("/api/chat/stream", json={"conversation_id": cid, "message": huge})

    seen_contents = [m.content for m in model.seen]
    assert "第一句" not in seen_contents  # 被预算挤出（若未扣除当前输入则会保留）
    assert seen_contents[1] == "第二句"  # 至少保留最近一轮（H2+A2）
