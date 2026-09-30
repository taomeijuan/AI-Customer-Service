from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> ChatOpenAI:
    """按配置构建 ChatOpenAI——四类上游（GPT/Claude/DeepSeek/Ollama）统一 OpenAI 协议直连。"""
    s = settings or get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        api_key=s.llm_api_key or "-",  # Ollama 等无密钥上游用占位符
        base_url=s.llm_base_url,
        temperature=s.llm_temperature,
        streaming=True,
    )
