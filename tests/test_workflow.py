from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.agent.workflow import build_workflow, to_legacy_itinerary
from src.domain.models import Briefing, ItineraryItem, Place, Slot, SlotReason, TripPreferences
from src.domain.schedule import assign_times


def it(sid, day, order, cat, name=None):
    return ItineraryItem(slot=Slot(slot_id=sid, day=day, order=order, category=cat),
                         place=Place(name=name, region="r", category=cat) if name else None)


def test_assign_times_day1_noon_and_moves():
    items = [it("d1-s1", 1, 1, "식당", "A"), it("d1-s2", 1, 2, "카페", "B"), it("d2-s1", 2, 1, "관광지", "C")]
    out = assign_times(items, {"d1-s2": (20, "버스 402")})
    assert (out[0].start, out[0].end) == ("12:00", "13:30")          # 1일차 12시, 식당 90분
    assert (out[1].move_min, out[1].start, out[1].end) == (20, "13:50", "14:50")
    assert out[1].move_detail == "버스 402"
    assert out[2].start == "10:00"                                    # 2일차 10시


def test_assign_times_skips_empty_slots_and_defaults_move():
    items = [it("d1-s1", 1, 1, "식당", "A"), it("d1-s2", 1, 2, "카페"), it("d1-s3", 1, 3, "관광지", "C")]
    out = assign_times(items, {})
    assert out[1].start is None
    assert out[2].move_min == 30 and out[2].start == "14:00"          # 경로 없으면 30분 가정


# --- 워크플로우 E2E (가짜 의존성) -------------------------------------------------
def doc(name, cat):
    return Document(page_content="x", metadata={"장소명": name, "지역": "부산광역시 중구", "카테고리": cat,
                                                "상세 주소": f"{name} 주소", "별점": 4.0, "리뷰": f"{name} 좋아요"})


class Emb:
    def embed_documents(self, texts):
        return [[0.0]] * len(texts)


class DB:
    embedding_function = Emb()

    def similarity_search_by_vector(self, v, k=4):
        return [doc(f"식당{i}", "식당") for i in range(3)] + [doc("카페0", "카페"), doc("명소0", "관광지")]


class LLM:
    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema):
        def run(_):
            self.calls += 1
            if schema is TripPreferences:
                return TripPreferences(keywords=["노포"], avoid=[])
            return Briefing(overview="좋은 여행", weather_tip="우산", reasons=[SlotReason(slot_id="d1-s1", reason="노포 리뷰")])
        return RunnableLambda(run)


async def route(a, b, mode="transit", departure_time=None):
    return {"duration_value": 900, "steps": ["[버스] 11"]}


REQ = {"destination": "부산 중구", "start_location": "부산역", "start_date": "2026-10-01", "days": 1,
       "group_type": "친구", "style": "맛집 탐방", "detail": "노포"}


async def test_workflow_end_to_end_with_two_llm_calls():
    llm = LLM()
    app = build_workflow(llm=llm, load_db=lambda: DB(), route_fn=route, weather_fn=lambda dest, dates: "맑음")
    out = await app.ainvoke({"request": REQ})
    items = out["items"]
    assert out["region"] == "부산광역시 중구"
    assert [i.place.name if i.place else None for i in items] == ["식당0", "카페0", "명소0", "식당1"]
    assert items[0].reason == "노포 리뷰" and items[1].move_min == 15
    assert llm.calls == 2
    assert "식당0" in out["final_text"] and "우산" in out["final_text"]
    legacy = to_legacy_itinerary(items)
    assert [x["type"] for x in legacy][:3] == ["식당", "move", "카페"]
    assert legacy[0]["address"] == "부산광역시 중구" and legacy[1]["duration_min"] == 15


async def test_workflow_survives_llm_failures():
    class Broken:
        def with_structured_output(self, schema):
            def boom(_):
                raise ValueError("bad json")
            return RunnableLambda(boom)
    app = build_workflow(llm=Broken(), load_db=lambda: DB(), route_fn=route, weather_fn=lambda d, t: "맑음")
    out = await app.ainvoke({"request": REQ})
    assert out["items"][0].place.name == "식당0"           # 키워드 없이도 검색·배정
    assert "식당0" in out["final_text"]                     # 브리핑 실패해도 일정은 출력


async def test_unresolvable_destination_uses_fallback_resolver():
    app = build_workflow(llm=LLM(), load_db=lambda: DB(), route_fn=route, weather_fn=lambda d, t: "",
                         resolve_fallback=lambda dest: _async("부산광역시 중구"))
    out = await app.ainvoke({"request": {**REQ, "destination": "중구"}})
    assert out["region"] == "부산광역시 중구"


async def _async(v):
    return v
