from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.core.llm import get_chat_model


def test_builds_chat_openai_from_settings():
    s = Settings(
        _env_file=None,
        llm_model="deepseek-chat",
        llm_base_url="https://api.deepseek.com/v1",
        llm_api_key="sk-test",
    )
    m = get_chat_model(s)
    assert isinstance(m, ChatOpenAI)
    assert m.model_name == "deepseek-chat"
    assert m.openai_api_base == "https://api.deepseek.com/v1"


def test_empty_key_gets_placeholder():
    s = Settings(_env_file=None, llm_model="qwen2.5:7b", llm_api_key="")
    m = get_chat_model(s)
    assert m.openai_api_key  # Ollama 无密钥时兜底占位符，openai client 不接受空串
