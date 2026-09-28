from datetime import date

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from eval.legacy_adapter import build_preference, initial_state, run_legacy_trip

TRIP = {"id": "t", "destination": "부산", "start_location": "부산역", "days": 2,
        "group_type": "친구", "style": "맛집 탐방", "detail": "노포 좋아요"}


def test_preference_matches_app_format():
    assert build_preference(TRIP) == "- 동행: 친구\n- 스타일: 맛집 탐방\n- 출발지: 부산역\n- 상세 요청: 노포 좋아요"


def test_initial_state_mirrors_trip_planner():
    s = initial_state(TRIP, date(2026, 10, 1))
    assert s["dates"] == "2026년 10월 01일부터 2일간"
    assert s["total_days"] == 2 and s["dialog_stage"] == "planning"
    assert s["style"] == s["preference"]          # 원본 앱의 동작 그대로
    assert isinstance(s["messages"][0], HumanMessage)
    assert "목적지: 부산" in s["messages"][0].content


class App:
    def __init__(self, result=None, exc=None):
        self.result, self.exc = result, exc

    async def ainvoke(self, state, config=None):
        if self.exc:
            raise self.exc
        return self.result


async def test_run_records_recursion_error_instead_of_crashing():
    r = await run_legacy_trip(App(exc=GraphRecursionError("limit")), TRIP, date(2026, 10, 1))
    assert r["ok"] is False and r["error"] == "GraphRecursionError" and r["itinerary"] == []


async def test_run_returns_itinerary_and_last_ai_message():
    res = {"itinerary": [{"name": "a", "type": "식당", "day": 1}],
           "messages": [HumanMessage("q"), AIMessage("계획표")]}
    r = await run_legacy_trip(App(result=res), TRIP, date(2026, 10, 1))
    assert r["ok"] and r["itinerary"][0]["name"] == "a" and r["final_message"] == "계획표"
