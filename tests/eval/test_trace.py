import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from eval.trace import AgentTracer, percentile, timed


@tool
def lookup(city: str) -> str:
    """도시 조회"""
    if city == "bad":
        raise ValueError("boom")
    return city


async def test_tracer_collects_tools_llm_and_errors():
    llm = FakeListChatModel(responses=["a", "b"])

    async def node(_):
        await llm.ainvoke("hi")
        await lookup.ainvoke({"city": "부산"})
        await lookup.ainvoke({"city": "부산"})      # 같은 인자 반복 → 중복 호출
        try:
            await lookup.ainvoke({"city": "bad"})
        except ValueError:
            pass
        return 1

    tr = AgentTracer()
    await RunnableLambda(node).ainvoke(0, config={"callbacks": [tr]})
    s = tr.summary()
    assert s["llm_calls"] == 1 and s["llm_time_s"] >= 0
    assert s["tool_calls"] == 3 and s["tool_errors"] == 1
    assert s["tool_error_rate"] == pytest.approx(1 / 3)
    assert s["redundant_tool_calls"] == 1
    assert s["tool_calls_by_name"] == {"lookup": 3}


async def test_timed_accumulates():
    acc = {"t": 0.0, "n": 0}

    async def f(x):
        return x * 2

    g = timed(f, acc)
    assert await g(2) == 4 and acc["n"] == 1 and acc["t"] >= 0


def test_percentile():
    assert percentile([1, 2, 3, 4, 100], 50) == 3
    assert percentile([1, 2, 3, 4, 100], 95) == 100
    assert percentile([], 50) is None


async def test_soft_errors_and_invalid_tool_calls():
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult
    import uuid

    @tool
    def finder(q: str) -> str:
        """장소 찾기"""
        return '{"name": "추천 장소 없음"}'

    tr = AgentTracer(known_tools={"finder"})
    await RunnableLambda(lambda _: finder.invoke({"q": "x"})).ainvoke(0, config={"callbacks": [tr]})
    msg = AIMessage(content="", tool_calls=[{"name": "finder", "args": {}, "id": "1"},
                                            {"name": "ghost_tool", "args": {}, "id": "2"}],
                    invalid_tool_calls=[{"name": "finder", "args": "{bad", "id": "3", "error": "json"}])
    rid = uuid.uuid4()
    await tr.on_chat_model_start({}, [], run_id=rid)
    await tr.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=rid)
    s = tr.summary()
    assert s["tool_soft_errors"] == 1
    assert s["tool_calls_requested"] == 3 and s["invalid_tool_calls"] == 2   # 파싱 실패 1 + 없는 도구 1
    assert s["invalid_tool_call_rate"] == pytest.approx(2 / 3)
