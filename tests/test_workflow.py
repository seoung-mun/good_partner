from langchain_core.runnables import RunnableLambda

from src.agent.workflow import build_workflow, to_legacy_itinerary
from src.domain.models import Briefing, ItineraryItem, Place, Slot, SlotReason, TripPreferences
from src.domain.schedule import assign_times
from tests.fakes import make_db, meta


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
def DB():
    return make_db([meta(f"식당{i}", "부산광역시 중구", "식당", f"식당{i} 좋아요") for i in range(3)]
                   + [meta("카페0", "부산광역시 중구", "카페", "카페0 좋아요"),
                      meta("명소0", "부산광역시 중구", "관광지", "명소0 좋아요")])


class LLM:
    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema, **kw):
        def run(_):
            self.calls += 1
            if schema is TripPreferences:
                return TripPreferences(keywords=["노포"], avoid=[])
            return Briefing(overview="좋은 여행", weather_tip="우산", reasons=[SlotReason(slot_id="d1-s1", reason="노포 리뷰")])
        return RunnableLambda(run)


async def no_geo(query):
    return None, None


async def route(a, b, mode="transit", departure_time=None):
    return {"duration_value": 900, "steps": ["[버스] 11"]}


REQ = {"destination": "부산 중구", "start_location": "부산역", "start_date": "2026-10-01", "days": 1,
       "group_type": "친구", "style": "맛집 탐방", "detail": "노포"}


async def test_workflow_end_to_end_with_two_llm_calls():
    llm = LLM()
    app = build_workflow(geocode_fn=no_geo, llm=llm, load_db=lambda: DB(), route_fn=route, weather_fn=lambda dest, dates: "맑음")
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
        def with_structured_output(self, schema, **kw):
            def boom(_):
                raise ValueError("bad json")
            return RunnableLambda(boom)
    app = build_workflow(geocode_fn=no_geo, llm=Broken(), load_db=lambda: DB(), route_fn=route, weather_fn=lambda d, t: "맑음")
    out = await app.ainvoke({"request": REQ})
    assert out["items"][0].place.name == "식당0"           # 키워드 없이도 검색·배정
    assert "식당0" in out["final_text"]                     # 브리핑 실패해도 일정은 출력


async def test_unresolvable_destination_uses_fallback_resolver():
    app = build_workflow(geocode_fn=no_geo, llm=LLM(), load_db=lambda: DB(), route_fn=route, weather_fn=lambda d, t: "",
                         resolve_fallback=lambda dest: _async("부산광역시 중구"))
    out = await app.ainvoke({"request": {**REQ, "destination": "중구"}})
    assert out["region"] == "부산광역시 중구"


async def _async(v):
    return v


# --- 단계별 실패는 StepError로 기록하고 서비스는 계속 ---------------------------------
def codes(out):
    return [e.code for e in out.get("errors", [])]


async def test_llm_failures_are_recorded_as_step_errors():
    class Broken:
        def with_structured_output(self, schema, **kw):
            return RunnableLambda(lambda _: (_ for _ in ()).throw(ValueError("bad json")))
    out = await build_workflow(geocode_fn=no_geo, llm=Broken(), load_db=lambda: DB(), route_fn=route, weather_fn=lambda d, t: "맑음").ainvoke({"request": REQ})
    assert codes(out) == ["LLM_OUTPUT_INVALID", "LLM_OUTPUT_INVALID"]
    assert "AI 응답을 해석하지 못해" in out["final_text"]


async def test_vector_store_failure_does_not_crash():
    def no_db():
        raise FileNotFoundError("review_faiss_mean")
    out = await build_workflow(geocode_fn=no_geo, llm=LLM(), load_db=no_db, route_fn=route, weather_fn=lambda d, t: "맑음").ainvoke({"request": REQ})
    assert "VECTOR_STORE_UNAVAILABLE" in codes(out)
    assert all(i.place is None for i in out["items"])
    assert "장소 데이터베이스를 불러오지 못했어요" in out["final_text"]


async def test_region_not_found_is_reported():
    out = await build_workflow(geocode_fn=no_geo, llm=LLM(), load_db=lambda: DB(), route_fn=route, weather_fn=lambda d, t: "",
                               resolve_fallback=lambda dest: _async("")).ainvoke({"request": {**REQ, "destination": "중구"}})
    assert "REGION_NOT_FOUND" in codes(out)
    assert "시/도와 시/군/구를 함께" in out["final_text"]


async def test_route_and_weather_failures_use_defaults():
    async def bad_route(a, b, mode="transit", departure_time=None):
        raise TimeoutError("maps timeout")
    out = await build_workflow(geocode_fn=no_geo, llm=LLM(), load_db=lambda: DB(), route_fn=bad_route,
                               weather_fn=lambda d, t: "오류: Forecast API 호출 중 문제 발생").ainvoke({"request": REQ})
    assert codes(out).count("EXTERNAL_SERVICE_FAILED") == 2          # Maps 1건(요약), 날씨 1건
    assert out["items"][1].move_min == 30                             # 경로 없으면 기본 이동 시간


async def test_llm_timeout_is_recorded_and_service_continues():
    import asyncio

    class Slow:
        def with_structured_output(self, schema, **kw):
            async def hang(_):
                await asyncio.sleep(10)
            return RunnableLambda(hang)
    app = build_workflow(geocode_fn=no_geo, llm=Slow(), load_db=lambda: DB(), route_fn=route, weather_fn=lambda d, t: "맑음", llm_timeout=0.05)
    out = await asyncio.wait_for(app.ainvoke({"request": REQ}), timeout=5)
    assert codes(out) == ["LLM_OUTPUT_INVALID", "LLM_OUTPUT_INVALID"]
    assert "TimeoutError" in out["errors"][0].detail
    assert out["items"][0].place.name == "식당0"


async def test_independent_steps_run_in_parallel():
    """날씨 ∥ 취향 LLM, 경로 ∥ 브리핑 LLM — 순차 합(1.2s)보다 확실히 짧아야 한다."""
    import asyncio
    import time

    class SlowLLM(LLM):
        def with_structured_output(self, schema, **kw):
            inner = super().with_structured_output(schema)

            async def run(x):
                await asyncio.sleep(0.3)
                return inner.invoke(x)
            return RunnableLambda(run)

    async def slow_route(a, b, mode="transit", departure_time=None):
        await asyncio.sleep(0.3)
        return {"duration_value": 600, "steps": []}

    def slow_weather(d, t):
        time.sleep(0.3)
        return "맑음"

    llm = SlowLLM()
    app = build_workflow(geocode_fn=no_geo, llm=llm, load_db=lambda: DB(), route_fn=slow_route, weather_fn=slow_weather)
    t0 = time.perf_counter()
    out = await app.ainvoke({"request": REQ})
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.75, elapsed                     # 병렬이면 ~0.6s (pref∥weather, route∥brief). 순차면 1.2s
    assert llm.calls == 2 and out["items"][0].reason == "노포 리뷰" and out["items"][1].move_min == 10


async def test_stream_plan_shows_itinerary_before_briefing():
    import asyncio
    from src.agent.workflow import stream_plan

    class SlowBriefLLM(LLM):
        def with_structured_output(self, schema, **kw):
            inner = super().with_structured_output(schema)

            async def run(x):
                if schema is not TripPreferences:
                    await asyncio.sleep(0.2)            # 브리핑이 경로보다 늦게 끝남
                return inner.invoke(x)
            return RunnableLambda(run)

    shown = []
    app = build_workflow(geocode_fn=no_geo, llm=SlowBriefLLM(), load_db=lambda: DB(), route_fn=route, weather_fn=lambda d, t: "맑음")
    final = await stream_plan(app, REQ, on_plan=shown.append)
    assert len(shown) == 1 and "식당0" in shown[0] and "12:00" in shown[0]
    assert "노포 리뷰" not in shown[0]                 # 임시 표시는 사유 없이
    assert "노포 리뷰" in final["final_text"]


async def test_locate_uses_start_location_as_anchor_and_keeps_alternatives():
    coords = {"식당0 주소": (36.0, 129.0), "식당1 주소": (35.001, 129.0), "식당2 주소": (35.3, 129.0),
              "부산광역시 중구 부산역": (35.0, 129.0)}

    async def geo(q):
        return coords.get(q, (None, None))
    app = build_workflow(geocode_fn=geo, llm=LLM(), load_db=lambda: DB(), route_fn=route,
                         weather_fn=lambda d, t: "맑음")
    out = await app.ainvoke({"request": REQ})
    first = next(i for i in out["items"] if i.slot.slot_id == "d1-s1")
    assert first.place.name == "식당1"                       # 검색 2위지만 출발지(부산역)에서 가장 가까움
    assert first.place.lat == 35.001
    assert "식당0" not in [p.name for p in first.alternatives[:1]]   # 110km 밖은 대안 맨 앞이 아님
