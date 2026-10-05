import pytest

from app.knowledge.milvus_store import MilvusStore

TEST_COLLECTION = "knowledge_test"


@pytest.fixture
def store():
    s = MilvusStore(uri="http://localhost:19530", collection=TEST_COLLECTION, dim=4)
    s.recreate()  # 测试先清
    yield s
    s.drop()


def test_upsert_search_roundtrip(store):
    store.upsert(
        [
            {
                "id": 1,
                "vector": [0.9, 0, 0, 0],
                "questions": "运费",
                "answer": "满99包邮",
                "category": "售后",
                "content_type": "policy",
            },
            {
                "id": 2,
                "vector": [0, 0.9, 0, 0],
                "questions": "发货",
                "answer": "48小时",
                "category": "物流",
                "content_type": "policy",
            },
        ]
    )
    hits = store.search([0.9, 0.01, 0, 0], top_k=2)
    assert hits[0]["id"] == 1
    assert hits[0]["entity"]["answer"] == "满99包邮"


def test_upsert_same_id_overwrites(store):
    store.upsert(
        [{"id": 1, "vector": [1, 0, 0, 0], "questions": "旧", "answer": "旧", "category": "c", "content_type": "t"}]
    )
    store.upsert(
        [{"id": 1, "vector": [0, 1, 0, 0], "questions": "新", "answer": "新", "category": "c", "content_type": "t"}]
    )
    hits = store.search([0, 1, 0, 0], top_k=5)
    assert len(hits) == 1 and hits[0]["entity"]["questions"] == "新"  # 同 id 覆盖=幂等


def test_score_threshold_filters(store):
    store.upsert(
        [{"id": 1, "vector": [0.9, 0, 0, 0], "questions": "运费", "answer": "满99包邮", "category": "c", "content_type": "t"}]
    )
    near = store.search([0.9, 0.05, 0, 0], top_k=3, score_threshold=0.8)
    far = store.search([0.9, 0.05, 0, 0], top_k=3, score_threshold=0.999)
    assert len(near) == 1 and far == []
