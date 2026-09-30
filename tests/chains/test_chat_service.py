from dataclasses import dataclass

from langchain.messages import HumanMessage

from app.chains.chat import ChatService


@dataclass
class FakeChunk:
    text: str


class FakeModel:
    def __init__(self, pieces):
        self.pieces = pieces
        self.seen = None

    async def astream(self, messages):
        self.seen = messages
        for p in self.pieces:
            yield FakeChunk(p)


async def test_streams_text_chunks_and_formats_messages():
    model = FakeModel(["你", "好", "呀"])
    svc = ChatService(model)
    got = [c async for c in svc.astream([HumanMessage("在吗")], "你好")]
    assert got == ["你", "好", "呀"]
    # messages 结构：system + history + human
    assert model.seen[0].type == "system"
    assert model.seen[1].content == "在吗"
    assert model.seen[-1].content == "你好"


async def test_collect_full_text():
    svc = ChatService(FakeModel(["a", "b"]))
    full = await svc.collect(svc.astream([], "hi"))
    assert full == "ab"
