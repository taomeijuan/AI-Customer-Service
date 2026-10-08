from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str | None = None  # OpenAI 兼容上游；None=官方 OpenAI
    llm_api_key: str = ""  # Ollama 等无密钥上游由工厂兜底占位符
    llm_model: str = ""
    llm_temperature: float = 0.7
    token_budget: int = 4000

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "ecom"
    mysql_password: str = "ecom123"
    mysql_database: str = "ecom_cs"
    tool_timeout: float = 3.0
    tool_retries: int = 1

    ollama_embed_base_url: str = "http://localhost:11434/v1"
    embed_model: str = "bge-m3"
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "knowledge"
    retrieval_top_k: int = 3
    mine_batch_size: int = 10

    rerank_api_base: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 10
    hybrid_candidates: int = 50
    rrf_k: int = 60
    # rerank relevance 标定（2026-10-08 实测 SiliconFlow/bge-reranker-v2-m3）：
    # 无关≈0.00；弱相关 0.19–0.23；正确相关 0.25–0.35；明显相关 ≥0.45。
    # 故此闸默认 0.25（区别于 ch03 COSINE 尺度的 0.45，两把尺子不可混用）。
    retrieval_low_conf_threshold: float = 0.25
    faq_tool_timeout: float = 30.0  # RAG 链（改写+混合检索+精排+生成）专用预算
    agent_max_steps: int = 6  # ReAct 循环步数/token 消耗上限（ch05）


@lru_cache
def get_settings() -> Settings:
    return Settings()
