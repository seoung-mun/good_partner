from src.llm_factory import llm_kwargs


def test_gpt5_has_no_temperature():
    assert "temperature" not in llm_kwargs("openai:gpt-5-mini")


def test_others_temperature_zero():
    assert llm_kwargs("ollama:qwen2.5:7b-instruct-q4_K_M")["temperature"] == 0
    assert llm_kwargs("openai:gpt-4.1-mini")["temperature"] == 0


def test_ollama_output_capped_and_timeout(monkeypatch):
    from src.llm_factory import llm_kwargs
    kw = llm_kwargs("ollama:qwen2.5:7b-instruct-q4_K_M")
    assert kw["num_predict"] > 0                 # 무한 생성 방지
    assert llm_kwargs("openai:gpt-5-mini")["timeout"] > 0


def test_ollama_model_stays_loaded():
    from src.llm_factory import llm_kwargs
    assert llm_kwargs("ollama:qwen2.5:7b-instruct-q4_K_M")["keep_alive"] == "30m"
