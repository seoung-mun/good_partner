from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate

from eval.instrument import LLMCallCounter, attach_counter


def test_counts_calls_through_prebuilt_chain():
    llm = FakeListChatModel(responses=["a", "b", "c"])
    chain = ChatPromptTemplate.from_template("{x}") | llm   # 계측 전에 만든 체인
    counter = LLMCallCounter()
    attach_counter(llm, counter)
    chain.invoke({"x": "1"})
    llm.invoke("2")
    assert counter.calls == 2
    snap = counter.snapshot()
    counter.reset()
    assert snap["llm_calls"] == 2 and counter.calls == 0


def test_no_cache_disables_maps_cache_too(monkeypatch):
    from eval.instrument import configure_caches
    monkeypatch.setenv("MAPS_CACHE", "1")   # 테스트 종료 시 원래 값으로 복원되도록 등록
    state = configure_caches(no_cache=True)
    import os
    assert os.environ["MAPS_CACHE"] == "0"
    assert state == {"llm_cache": False, "maps_cache": False}
