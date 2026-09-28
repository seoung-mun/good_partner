"""LLM 생성. provider는 LLM_MODEL(provider:model) 설정으로만 결정한다."""
import os

from langchain.chat_models import init_chat_model

OLLAMA_MAX_TOKENS = 1024     # 구조화 출력이 공백·반복으로 끝나지 않는 경우 방지
API_TIMEOUT_S = 60


def llm_kwargs(model: str) -> dict:
    """provider별 생성 옵션. gpt-5 계열은 temperature 미지원."""
    if model.startswith("ollama:"):
        return {"temperature": 0, "num_predict": OLLAMA_MAX_TOKENS}
    kw = {"timeout": API_TIMEOUT_S, "max_retries": 1}
    return kw if model.startswith("openai:gpt-5") else {**kw, "temperature": 0}


def make_llm(model: str | None = None):
    """설정된 모델로 chat model 인스턴스를 만든다."""
    model = model or os.environ["LLM_MODEL"]
    return init_chat_model(model, **llm_kwargs(model))
