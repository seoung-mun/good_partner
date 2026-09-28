from src.llm_factory import llm_kwargs


def test_gpt5_has_no_temperature():
    assert llm_kwargs("openai:gpt-5-mini") == {}


def test_others_temperature_zero():
    assert llm_kwargs("ollama:qwen2.5:7b-instruct-q4_K_M") == {"temperature": 0}
    assert llm_kwargs("openai:gpt-4.1-mini") == {"temperature": 0}
