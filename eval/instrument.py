from langchain_core.callbacks import BaseCallbackHandler


class LLMCallCounter(BaseCallbackHandler):
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = self.input_tokens = self.output_tokens = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.calls += 1

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.calls += 1

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)

    def snapshot(self) -> dict:
        return {"llm_calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens}


def attach_counter(llm, counter: LLMCallCounter) -> None:
    llm.callbacks = [*(llm.callbacks or []), counter]


def enable_llm_cache(path: str = ".cache/llm.sqlite") -> None:
    from pathlib import Path
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=path))


def configure_caches(no_cache: bool) -> dict:
    """공식 측정(--no-cache)은 LLM·Maps 캐시를 모두 끈다."""
    import os
    if no_cache:
        os.environ["MAPS_CACHE"] = "0"
    else:
        enable_llm_cache()
    return {"llm_cache": not no_cache, "maps_cache": os.getenv("MAPS_CACHE", "1") != "0"}
