from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str | None = None  # OpenAI 兼容上游；None=官方 OpenAI
    llm_api_key: str = ""  # Ollama 等无密钥上游由工厂兜底占位符
    llm_model: str = ""
    llm_temperature: float = 0.7
    token_budget: int = 4000


@lru_cache
def get_settings() -> Settings:
    return Settings()
