from app.core.config import Settings, get_settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("TOKEN_BUDGET", raising=False)
    s = Settings(_env_file=None)
    assert s.llm_model == ""  # 允许空，启动健康检查前不强制
    assert s.token_budget == 4000
    assert s.llm_temperature == 0.7


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    s = Settings(_env_file=None)
    assert s.llm_model == "deepseek-chat"
    assert s.llm_base_url == "https://api.deepseek.com/v1"


def test_get_settings_cached(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "m1")
    get_settings.cache_clear()
    assert get_settings().llm_model == "m1"
    monkeypatch.setenv("LLM_MODEL", "m2")
    assert get_settings().llm_model == "m1"  # 缓存生效
    get_settings.cache_clear()
