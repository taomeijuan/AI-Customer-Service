from pymilvus import DataType, MilvusClient

_OUTPUT_FIELDS = ["questions", "answer", "category", "content_type"]


class MilvusStore:
    """Milvus 集合封装：主键 = MySQL chunk.id（双写对齐锚点），upsert 即幂等覆盖。"""

    def __init__(self, uri: str, collection: str, dim: int = 1024) -> None:
        self._client = MilvusClient(uri)
        self.collection = collection
        self.dim = dim
        self._ensured = False

    def ensure_collection(self) -> None:
        if self._ensured or self._client.has_collection(self.collection):
            self._ensured = True
            return
        schema = self._client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("questions", DataType.VARCHAR, max_length=2048)
        schema.add_field("answer", DataType.VARCHAR, max_length=65535)
        schema.add_field("category", DataType.VARCHAR, max_length=512)
        schema.add_field("content_type", DataType.VARCHAR, max_length=32)
        # pymilvus 3.x：schema 路径需显式建索引再 load
        self._client.create_collection(self.collection, schema=schema)
        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        self._client.create_index(self.collection, index_params=index_params)
        self._client.load_collection(self.collection)
        self._wait_loaded()

    def recreate(self) -> None:
        if self._client.has_collection(self.collection):
            self._client.drop_collection(self.collection)
            self._wait_gone()
        self._ensured = False
        self.ensure_collection()

    def _wait_gone(self, timeout: float = 10.0) -> None:
        """drop 是异步生效的：同名重建前轮询等它消失（否则偶发竞争）。"""
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._client.has_collection(self.collection):
                return
            time.sleep(0.1)

    def _wait_loaded(self, timeout: float = 30.0) -> None:
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self._client.get_load_state(self.collection)
            if state and state.get("state") is not None and "Loaded" in str(state):
                return
            time.sleep(0.2)

    def drop(self) -> None:
        if self._client.has_collection(self.collection):
            self._client.drop_collection(self.collection)

    def upsert(self, rows: list[dict]) -> None:
        """行: {id, vector, questions, answer, category, content_type}；同 id 覆盖。"""
        self.ensure_collection()
        self._client.upsert(self.collection, rows)

    def search(
        self,
        vector: list[float],
        top_k: int = 3,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """COSINE 相似度 Top-K；score_threshold 过滤低分（distance 即相似度）。"""
        self.ensure_collection()
        results = self._client.search(
            self.collection,
            data=[vector],
            limit=top_k,
            output_fields=_OUTPUT_FIELDS,
            consistency_level="Strong",  # 写后立读可见（upsert→search 竞态）
        )
        hits = []
        for hit in results[0]:
            if score_threshold is not None and hit["distance"] < score_threshold:
                continue
            hits.append(
                {"id": hit["id"], "distance": hit["distance"], "entity": hit["entity"]}
            )
        return hits

    def count(self) -> int:
        self.ensure_collection()
        return self._client.query(
            self.collection, filter="id >= 0", output_fields=["id"]
        ).__len__()


class LazyMilvusStore:
    """惰性包装：Milvus 未启动时应用仍可启动（在线检索/建库首次使用时才连接）。

    MilvusClient 构造即建连，若在 create_app 期直接实例化，Milvus 宕机会拖死
    整个应用（含与向量库无关的 ch01/ch02 功能）。惰性化后失败被限制在单次工具调用内，
    由 ToolRegistry 包装为 {ok: False} 回灌。
    """

    def __init__(self, uri: str, collection: str, dim: int = 1024) -> None:
        self._kwargs = {"uri": uri, "collection": collection, "dim": dim}
        self._store: MilvusStore | None = None

    def _get(self) -> MilvusStore:
        if self._store is None:
            self._store = MilvusStore(**self._kwargs)
        return self._store

    @property
    def collection(self) -> str:
        return self._kwargs["collection"]

    def ensure_collection(self) -> None:
        self._get().ensure_collection()

    def recreate(self) -> None:
        self._get().recreate()

    def drop(self) -> None:
        self._get().drop()

    def upsert(self, rows: list[dict]) -> None:
        self._get().upsert(rows)

    def search(
        self,
        vector: list[float],
        top_k: int = 3,
        score_threshold: float | None = None,
    ) -> list[dict]:
        return self._get().search(vector, top_k, score_threshold)
