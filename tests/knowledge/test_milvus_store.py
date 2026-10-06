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
                "text": "运费 满99包邮",
                "questions": "运费",
                "answer": "满99包邮",
                "category": "售后",
                "content_type": "policy",
            },
            {
                "id": 2,
                "vector": [0, 0.9, 0, 0],
                "text": "发货 48小时",
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
        [{"id": 1, "vector": [1, 0, 0, 0], "text": "旧", "questions": "旧", "answer": "旧", "category": "c", "content_type": "t"}]
    )
    store.upsert(
        [{"id": 1, "vector": [0, 1, 0, 0], "text": "新", "questions": "新", "answer": "新", "category": "c", "content_type": "t"}]
    )
    hits = store.search([0, 1, 0, 0], top_k=5)
    assert len(hits) == 1 and hits[0]["entity"]["questions"] == "新"  # 同 id 覆盖=幂等


def test_score_threshold_filters(store):
    store.upsert(
        [{"id": 1, "vector": [0.9, 0, 0, 0], "text": "运费", "questions": "运费", "answer": "满99包邮", "category": "c", "content_type": "t"}]
    )
    near = store.search([0.9, 0.05, 0, 0], top_k=3, score_threshold=0.8)
    far = store.search([0.9, 0.05, 0, 0], top_k=3, score_threshold=0.999)
    assert len(near) == 1 and far == []


def test_bm25_fulltext_hit(store):
    """ch04：BM25 全文检索按词面命中（含型号类精确 token）。"""
    store.upsert(
        [
            {"id": 1, "vector": [0.1, 0, 0, 0], "text": "空气炸锅 AF-102 的预约定时设置", "questions": "AF-102 预约", "answer": "长按预约键", "category": "FAQ", "content_type": "faq"},
            {"id": 2, "vector": [0, 0.1, 0, 0], "text": "耳机 EB-55 连接电脑", "questions": "EB-55 连电脑", "answer": "用随机 USB 接收器", "category": "FAQ", "content_type": "faq"},
        ]
    )
    hits = store.search_text("AF-102 预约", top_k=2)
    assert len(hits) >= 1 and hits[0]["id"] == 1  # 词面命中的 AF-102 排第一（零重叠文档本就不返回）


def test_hybrid_search_fuses_both(store):
    """ch04：dense + BM25 两路 RRF 融合，两路各自命中的块都出现。"""
    store.upsert(
        [
            {"id": 1, "vector": [0.95, 0.05, 0, 0], "text": "邮费与包邮规则 满99元包邮", "questions": "邮费", "answer": "满99包邮", "category": "售后", "content_type": "policy"},
            {"id": 2, "vector": [0.1, 0.9, 0, 0], "text": "退货运费承担 7天内", "questions": "退货运费", "answer": "买家承担", "category": "售后", "content_type": "policy"},
        ]
    )
    hits = store.hybrid_search([0.9, 0.2, 0, 0], "邮费 运费承担", top_k=2)
    ids = {h["id"] for h in hits}
    assert ids == {1, 2}  # dense 命中 1、BM25 词面命中 2，融合后都在
