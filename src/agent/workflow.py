"""초안 일정 생성 워크플로우 (LangGraph). 결정적 단계는 코드, LLM은 취향 파싱·브리핑 2회만.

parse_request → search → (plan_routes ∥ fetch_weather) → compose
"""
import asyncio
from datetime import date
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.chains import build_briefing_chain, build_preference_chain, format_plan_for_briefing
from src.agent.render import render_markdown, request_text
from src.domain.models import ItineraryItem, Slot, TripPreferences
from src.domain.region import resolve_destination
from src.domain.schedule import assign_places, assign_times, build_slots
from src.retrieval.retriever import search_candidates


class PlanState(TypedDict, total=False):
    request: dict            # destination, start_location, start_date(YYYY-MM-DD), days, group_type, style, detail
    region: str
    preferences: TripPreferences
    slots: list[Slot]
    items: list[ItineraryItem]
    weather: str
    final_text: str


def _where(item: ItineraryItem) -> str:
    p = item.place
    return p.address or f"{p.region} {p.name}"


async def reschedule(items: list[ItineraryItem], route_fn) -> list[ItineraryItem]:
    """같은 날 연속한 장소 쌍의 경로를 병렬 조회(I/O)하고 시간표를 다시 계산."""
    pairs, prev = [], None
    for it in items:
        if prev is not None and prev.slot.day != it.slot.day:
            prev = None
        if it.place is not None:
            if prev is not None:
                pairs.append((it.slot.slot_id, prev, it))
            prev = it
    results = await asyncio.gather(*[route_fn(_where(a), _where(b), mode="transit") for _, a, b in pairs],
                                   return_exceptions=True)
    legs = {sid: (r["duration_value"] // 60, " ➡️ ".join(r.get("steps", [])))
            for (sid, _, _), r in zip(pairs, results) if isinstance(r, dict)}
    return assign_times([i.model_copy(update={"move_min": None, "move_detail": None}) for i in items], legs)


def to_legacy_itinerary(items: list[ItineraryItem]) -> list[dict]:
    """평가·PDF가 읽는 기존 itinerary 형식(type/name/address/day + move 항목)으로 변환."""
    out = []
    for it in items:
        if it.place is None:
            continue
        if it.move_min is not None:
            out.append({"type": "move", "duration_min": it.move_min, "day": it.slot.day,
                        "transport_detail": it.move_detail or "", "transport": it.move_detail or "이동",
                        "duration_text": f"약 {it.move_min}분"})
        out.append({"type": it.place.category, "name": it.place.name, "address": it.place.region,
                    "day": it.slot.day, "category": it.place.category, "slot_id": it.slot.slot_id, "start": it.start, "end": it.end,
                    "description": it.reason, "reviews": it.place.reviews})
    return out


def build_workflow(llm=None, load_db=None, route_fn=None, weather_fn=None, resolve_fallback=None):
    """의존성 주입 가능 — 기본값은 앱 설정(LLM, FAISS, Google Maps, OpenWeatherMap)."""
    if llm is None:
        from src.config import LLM as llm
    if load_db is None:
        from src.retrieval.store import load_faiss_index as load_db
    if route_fn is None:
        from src.services.maps import get_detailed_route as route_fn
    if weather_fn is None:
        from src.services.weather import forecast_summary as weather_fn
    if resolve_fallback is None:
        from src.services.maps import resolve_admin_region
        resolve_fallback = lambda dest: resolve_admin_region(dest, dest)  # noqa: E731
    pref_chain, brief_chain = build_preference_chain(llm), build_briefing_chain(llm)

    async def parse_request(state: PlanState):
        req = state["request"]
        resolved = resolve_destination(req["destination"])
        region = " ".join(x for x in resolved if x) if resolved else (await resolve_fallback(req["destination"])).strip()
        try:
            prefs = await pref_chain.ainvoke({"group_type": req["group_type"], "style": req["style"],
                                              "detail": req.get("detail", "")})
        except Exception as e:  # 취향 파싱 실패 → 키워드 없이 기본 쿼리로 진행
            print(f"취향 파싱 실패: {e}")
            prefs = TripPreferences(keywords=[], avoid=[])
        return {"region": region, "preferences": prefs, "slots": build_slots(req["days"])}

    async def search(state: PlanState):
        cats = list(dict.fromkeys(s.category for s in state["slots"]))
        cands = await asyncio.to_thread(search_candidates, load_db(), state["region"], cats,
                                        state["preferences"].keywords)
        picks = assign_places(state["slots"], cands)
        return {"items": [ItineraryItem(slot=s, place=picks[s.slot_id]) for s in state["slots"]]}

    async def plan_routes(state: PlanState):
        return {"items": await reschedule(state["items"], route_fn)}

    async def fetch_weather(state: PlanState):
        req = state["request"]
        dates = date.fromisoformat(req["start_date"]).strftime("%Y년 %m월 %d일")
        try:
            return {"weather": await asyncio.to_thread(weather_fn, req["destination"], dates)}
        except Exception as e:
            print(f"날씨 조회 실패: {e}")
            return {"weather": ""}

    async def compose(state: PlanState):
        req, items = state["request"], state["items"]
        try:
            briefing = await brief_chain.ainvoke({"request": request_text(req), "weather": state.get("weather") or "정보 없음",
                                                  "plan": format_plan_for_briefing(items)})
            reasons = {r.slot_id: r.reason for r in briefing.reasons}
            items = [i.model_copy(update={"reason": reasons.get(i.slot.slot_id, "")}) for i in items]
        except Exception as e:  # 브리핑 실패해도 일정은 결정적으로 렌더
            print(f"브리핑 실패: {e}")
            briefing = None
        return {"items": items, "final_text": render_markdown(req, items, briefing, state.get("weather", ""))}

    g = StateGraph(PlanState)
    for name, fn in [("parse_request", parse_request), ("search", search), ("plan_routes", plan_routes),
                     ("fetch_weather", fetch_weather), ("compose", compose)]:
        g.add_node(name, fn)
    g.add_edge(START, "parse_request")
    g.add_edge("parse_request", "search")
    g.add_edge("search", "plan_routes")
    g.add_edge("search", "fetch_weather")
    g.add_edge("plan_routes", "compose")
    g.add_edge("fetch_weather", "compose")
    g.add_edge("compose", END)
    return g.compile()
