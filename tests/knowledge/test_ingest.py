import pytest

from app.knowledge.ingest import ingest_pending
from app.knowledge.repository import KnowledgeRepo


class FakeEmbedder:
    def __init__(self, fail_first=0):
        self.fail_first = fail_first
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("模拟中断")
        return [[0.1, 0.2] for _ in texts]


class FakeMilvus:
    def __init__(self):
        self.rows: dict[int, dict] = {}
        self.existing_ids: set | None = None  # None=与 rows 同步

    def all_ids(self):
        if self.existing_ids is not None:
            return self.existing_ids
        return set(self.rows)

    def upsert(self, rows):
        for r in rows:
            self.rows[r["id"]] = r
            if self.existing_ids is not None:
                self.existing_ids.add(r["id"])


@pytest.fixture
def repo(db_session):
    return KnowledgeRepo(db_session)


async def test_ingest_backfills_vector_id(repo, db_session):
    """验收2核心：pending → Milvus → 回填 done。"""
    ids = await repo.upsert_chunks(
        [{"category": "c", "questions": "q", "answer": "a", "content_type": "policy"}]
    )
    milvus = FakeMilvus()
    done = await ingest_pending(repo, FakeEmbedder(), milvus)
    assert done == 1
    k = await repo.get(ids[0])
    assert k.vectorize_status == "done"
    assert k.vector_id == str(ids[0])
    assert milvus.rows[ids[0]]["questions"] == "q"  # 向量化文本外，Milvus 副本可检索


async def test_resume_after_milvus_written_before_backfill(repo, db_session):
    """评审 Minor#12①：Milvus 已写入但 MySQL 未回填的分支——重跑 upsert 同 id 覆盖补齐。"""
    ids = await repo.upsert_chunks(
        [{"category": "c", "questions": "q", "answer": "a", "content_type": "policy"}]
    )
    milvus = FakeMilvus()
    milvus.upsert([{"id": ids[0], "vector": [0.0], "questions": "q", "answer": "a",
                    "category": "c", "content_type": "policy"}])  # 模拟：Milvus 已写
    k = await repo.get(ids[0])
    assert k.vectorize_status == "pending"  # 但 MySQL 未回填（中断点）
    done = await ingest_pending(repo, FakeEmbedder(), milvus)
    assert done == 1
    k = await repo.get(ids[0])
    assert k.vectorize_status == "done" and k.vector_id == str(ids[0])


async def test_resume_picks_up_pending(repo):
    """中断重跑：第一次 embed 后崩（Milvus 未写），重跑补齐。"""
    await repo.upsert_chunks(
        [
            {"category": "c", "questions": "q1", "answer": "a1", "content_type": "policy"},
            {"category": "c", "questions": "q2", "answer": "a2", "content_type": "policy"},
        ]
    )
    milvus = FakeMilvus()
    with pytest.raises(RuntimeError):
        await ingest_pending(repo, FakeEmbedder(fail_first=1), milvus)
    assert milvus.rows == {}  # 第一批即崩，什么都没写
    done = await ingest_pending(repo, FakeEmbedder(), milvus)
    assert done == 2  # 重跑全部补齐


async def test_doc_rebuild_reuses_existing_chunk(repo):
    """文档级幂等：内容相同复用已有行，不重复建。"""
    chunk = {"category": "c", "questions": "q", "answer": "a", "content_type": "policy"}
    ids1 = await repo.upsert_chunks([chunk])
    ids2 = await repo.upsert_chunks([dict(chunk)])
    assert len(ids2) == 1 and ids2[0] == ids1[0]


async def test_link_neighbors(repo):
    """prev/next 指针回填。"""
    ids = await repo.upsert_chunks(
        [
            {"category": "c", "questions": "q1", "answer": "a1", "content_type": "policy"},
            {"category": "c", "questions": "q2", "answer": "a2", "content_type": "policy"},
            {"category": "c", "questions": "q3", "answer": "a3", "content_type": "policy"},
        ]
    )
    await repo.link_neighbors(ids)
    first, mid, last = [await repo.get(i) for i in ids]
    assert first.next_chunk_id == ids[1] and first.prev_chunk_id is None
    assert mid.prev_chunk_id == ids[0] and mid.next_chunk_id == ids[2]
    assert last.prev_chunk_id == ids[1] and last.next_chunk_id is None


async def test_reconcile_backfills_missing_milvus_vectors(repo):
    """验收2扩展（ch04 发现）：集合重建后 MySQL 全 done 但 Milvus 缺向量 → 对账补齐。"""
    await repo.upsert_chunks(
        [
            {"category": "c", "questions": "q1", "answer": "a1", "content_type": "policy"},
            {"category": "c", "questions": "q2", "answer": "a2", "content_type": "policy"},
        ]
    )
    milvus = FakeMilvus()
    milvus.existing_ids = set()  # 模拟空集合（重建后）
    done = await ingest_pending(repo, FakeEmbedder(), milvus)
    assert done == 2  # 两块都补写
    assert set(milvus.rows) == {1, 2}
