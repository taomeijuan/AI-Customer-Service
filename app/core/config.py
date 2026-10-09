from functools import lru_cache

from pydantic import AliasChoices, Field
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
    # ch07 改名 RERANK_TOP_K（旧名兼容），且成为精排取数与证据预算的单口径
    rerank_top_k: int = Field(default=10, validation_alias=AliasChoices("RERANK_TOP_K", "RERANK_TOP_N"))
    hybrid_candidates: int = 50
    rrf_k: int = 60
    # rerank relevance 标定（2026-10-08 实测 SiliconFlow/bge-reranker-v2-m3）：
    # 无关≈0.00；弱相关 0.19–0.23；正确相关 0.25–0.35；明显相关 ≥0.45。
    # 故此闸默认 0.25（区别于 ch03 COSINE 尺度的 0.45，两把尺子不可混用）。
    retrieval_low_conf_threshold: float = 0.25
    faq_tool_timeout: float = 30.0  # RAG 链（改写+混合检索+精排+生成）专用预算
    max_agent_steps: int = Field(
        default=6, validation_alias=AliasChoices("MAX_AGENT_STEPS", "AGENT_MAX_STEPS")
    )  # ReAct 循环步数/token 消耗上限（ch05；ch07 改名对齐预算口径）

    # ---- ch07 上下文预算：窗口倒推的输入项与固定开销预留（详见 context-design §5）----
    model_context_window: int = 64000  # 模型窗口（DeepSeek 64K）
    max_output_tokens: int = 2000  # 输出预留
    max_user_input_tokens: int = 2000  # 当前轮用户输入预留
    tool_result_max_tokens: int = 1200  # 单条工具结果预留（峰值项乘数）
    system_reserve: int = 2000  # X：system 人设+红线+工具定义 预留
    evidence_per_item_reserve: int = 400  # y：单条检索证据预留（×RERANK_TOP_K）
    summary_inject_reserve: int = 500  # Z：梗概投影注入预留
    safety_reserve: int = 250  # S：安全余量
    keep_turns: int = 24  # 想留住的轮数（期望侧分子）
    turn_steady_tokens: int = 400  # 每轮稳态占用（期望侧乘数）
    assistant_trunc_chars: int = 80  # 层2 规则：assistant 只留开头 N 字


@lru_cache
def get_settings() -> Settings:
    return Settings()
