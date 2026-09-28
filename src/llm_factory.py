import os

from langchain.chat_models import init_chat_model


def llm_kwargs(model: str) -> dict:
    # gpt-5 계열은 temperature를 지원하지 않음
    return {} if model.startswith("openai:gpt-5") else {"temperature": 0}


def make_llm(model: str | None = None):
    model = model or os.environ["LLM_MODEL"]
    return init_chat_model(model, **llm_kwargs(model))
