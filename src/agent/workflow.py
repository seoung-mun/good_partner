"""초안 일정 생성 워크플로우 (LangGraph).

    resolve_region ─┬─ extract_preferences ─ search ─┬─ plan_routes ─┐
                    └─ fetch_weather ────────────────┴─ brief ───────┴─ render

임계 경로: 지역 해석 → 취향 LLM → 검색 → max(경로, 브리핑 LLM) → 렌더. 날씨·경로는 LLM 대기와 겹쳐 실행된다.

결정적 단계(지역 해석·슬롯·검색·배정·시간표·렌더)는 코드가, LLM은 취향 파싱과 브리핑 2회만 담당한다.
각 단계의 실패는 예외로 전파하지 않고 StepError로 상태에 기록한 뒤 대체 동작으로 계속 진행한다.

비동기 경계: 네트워크 I/O(LLM, Maps)는 await, CPU 작업(임베딩·FAISS)과 동기 HTTP(날씨)는 asyncio.to_thread.
"""
import asyncio
import operator
import os
from datetime import date
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.chains import build_briefing_chain, build_preference_chain, format_plan_for_briefing
from src.agent.render import render_markdown, request_text
from src.domain.models import Briefing, ItineraryItem, Slot, StepError, TripPreferences
from src.domain.region import resolve_destination
from src.domain.schedule import assign_places, assign_times, build_slots
from src.errors import ExternalServiceError, LLMOutputError, RegionNotFoundError, VectorStoreUnavailableError
from src.logging_config import get_logger
from src.retrieval.retriever import search_candidates

log = get_logger(__name__)


class PlanState(TypedDict, total=False):
    """워크플로우 상태. errors는 여러 단계(병렬 포함)가 추가만 하므로 리스트 합치기 reducer."""

    request: dict            # destination, start_location, start_date(YYYY-MM-DD), days, group_type, style, detail
    region: str              # "부산광역시 중구" (해석 실패 시 "")
    preferences: TripPreferences
    slots: list[Slot]
    items: list[ItineraryItem]
    weather: str
    briefing: Briefing | None
    final_text: str
    errors: Annotated[list[StepError], operator.add]


def _where(item: ItineraryItem) -> str:
    """경로 검색용 위치 문자열: 상세 주소 우선, 없으면 '지역 장소명'."""
    p = item.place
    return p.address or f"{p.region} {p.name}"


async def reschedule(items: list[ItineraryItem], route_fn) -> tuple[list[ItineraryItem], list[StepError]]:
    """같은 날 연속한 장소 쌍의 경로를 병렬 조회하고 시간표를 다시 계산. 실패한 구간은 기본 이동 시간."""
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
    legs, failures = {}, []
    for (sid, _, _), r in zip(pairs, results):
        if isinstance(r, dict):
            legs[sid] = (r["duration_value"] // 60, " ➡️ ".join(r.get("steps", [])))
        elif isinstance(r, Exception):
            failures.append(f"{sid}: {type(r).__name__}: {r}")
    errors = []
    if failures:
        log.warning("경로 조회 실패 %d/%d건: %s", len(failures), len(pairs), failures[:3])
        errors.append(StepError.from_error("plan_routes", ExternalServiceError("Google Maps", "; ".join(failures[:3]))))
    reset = [i.model_copy(update={"move_min": None, "move_detail": None}) for i in items]
    return assign_times(reset, legs), errors


def to_legacy_itinerary(items: list[ItineraryItem]) -> list[dict]:
    """평가·PDF가 읽는 itinerary 형식(type/name/address/day + move 항목)으로 변환."""
    out = []
    for it in items:
        if it.place is None:
            continue
        if it.move_min is not None:
            out.append({"type": "move", "duration_min": it.move_min, "day": it.slot.day,
                        "transport_detail": it.move_detail or "", "transport": it.move_detail or "이동",
                        "duration_text": f"약 {it.move_min}분"})
        out.append({"type": it.place.category, "name": it.place.name, "address": it.place.region,
                    "day": it.slot.day, "category": it.place.category, "slot_id": it.slot.slot_id,
                    "start": it.start, "end": it.end, "description": it.reason, "reviews": it.place.reviews})
    return out


def build_workflow(llm=None, load_db=None, route_fn=None, weather_fn=None, resolve_fallback=None,
                   llm_timeout: float | None = None):
    """초안 생성 그래프를 만든다. 의존성은 주입 가능(테스트), 기본값은 앱 설정(LLM·FAISS·Maps·날씨).

    llm_timeout: LLM 호출 1회 제한 시간(초, 기본 LLM_TIMEOUT_S=90). 넘기면 StepError로 기록하고 대체 동작.
    """
    llm_timeout = llm_timeout or float(os.getenv("LLM_TIMEOUT_S", "90"))
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

    async def resolve_region(state: PlanState) -> dict:
        """목적지 → 행정구역(결정적, 모호하면 Maps) + 슬롯 생성. LLM 없음."""
        req, errors = state["request"], []
        resolved = resolve_destination(req["destination"])
        if resolved:
            region = " ".join(x for x in resolved if x)
        else:
            try:
                region = (await resolve_fallback(req["destination"]) or "").strip()
            except Exception as e:
                log.warning("목적지 Maps 해석 실패: %s", e)
                region = ""
            if not region:
                errors.append(StepError.from_error("resolve_region", RegionNotFoundError(req["destination"])))
        log.info("지역 해석: %s → %s", req["destination"], region or "?")
        return {"region": region, "slots": build_slots(req["days"]), "errors": errors}

    async def extract_preferences(state: PlanState) -> dict:
        """취향 키워드 추출 (LLM 1회, 제한 시간). 실패 시 키워드 없이 진행."""
        req = state["request"]
        try:
            prefs = await asyncio.wait_for(pref_chain.ainvoke({"group_type": req["group_type"], "style": req["style"],
                                                               "detail": req.get("detail", "")}), llm_timeout)
            log.info("취향 키워드: %s", prefs.keywords)
            return {"preferences": prefs}
        except Exception as e:
            log.warning("취향 파싱 실패 → 키워드 없이 진행: %s", e)
            return {"preferences": TripPreferences(keywords=[], avoid=[]),
                    "errors": [StepError.from_error("extract_preferences", LLMOutputError("preference", e))]}

    async def search(state: PlanState) -> dict:
        """모든 슬롯 쿼리를 한 번에 배치 검색해 슬롯에 배정. 지역 미확정·인덱스 실패 시 빈 슬롯."""
        slots = state["slots"]
        empty = {"items": [ItineraryItem(slot=s) for s in slots]}
        if not state.get("region"):
            return empty
        try:
            db = load_db()
            if db is None:
                raise FileNotFoundError("load_faiss_index returned None")
        except Exception as e:
            log.error("벡터 인덱스 로드 실패: %s", e)
            return {**empty, "errors": [StepError.from_error("search", VectorStoreUnavailableError(str(e)))]}
        cats = list(dict.fromkeys(s.category for s in slots))
        cands = await asyncio.to_thread(search_candidates, db, state["region"], cats, state["preferences"].keywords)
        picks = assign_places(slots, cands)
        log.info("검색 완료: 슬롯 %d개 중 %d개 배정", len(slots), sum(p is not None for p in picks.values()))
        return {"items": [ItineraryItem(slot=s, place=picks[s.slot_id]) for s in slots]}

    async def plan_routes(state: PlanState) -> dict:
        """연속 장소 간 대중교통 경로(병렬)와 시간표."""
        items, errors = await reschedule(state["items"], route_fn)
        return {"items": items, "errors": errors}

    async def fetch_weather(state: PlanState) -> dict:
        """여행 시작일 날씨 요약. 서비스는 실패를 '오류:'/'정보:' 문자열로 돌려주므로 함께 처리."""
        req = state["request"]
        dates = date.fromisoformat(req["start_date"]).strftime("%Y년 %m월 %d일")
        try:
            place = state.get("region") or req["destination"]   # 정식 지역명 → 날씨 서비스가 좌표로 해석
            text = await asyncio.to_thread(weather_fn, place, dates)
            if not text or text.startswith(("오류", "정보:")):
                raise RuntimeError(text or "empty")
            return {"weather": text}
        except Exception as e:
            log.warning("날씨 조회 실패: %s", e)
            return {"weather": "", "errors": [StepError.from_error("fetch_weather", ExternalServiceError("날씨", str(e)))]}

    async def brief(state: PlanState) -> dict:
        """근거 리뷰 기반 브리핑 (LLM 1회). 검색 결과만 필요하므로 경로 계산과 병렬로 실행."""
        req, items = state["request"], state["items"]
        if not any(i.place for i in items):
            return {"briefing": None}
        try:
            briefing = await asyncio.wait_for(brief_chain.ainvoke({"request": request_text(req),
                                                                   "weather": state.get("weather") or "정보 없음",
                                                                   "plan": format_plan_for_briefing(items)}), llm_timeout)
            return {"briefing": briefing}
        except Exception as e:
            log.warning("브리핑 실패 → 사유 없이 렌더: %s", e)
            return {"briefing": None, "errors": [StepError.from_error("brief", LLMOutputError("briefing", e))]}

    async def render(state: PlanState) -> dict:
        """시간표가 계산된 일정에 브리핑 사유를 합쳐 마크다운으로. LLM 없음."""
        items, briefing = state["items"], state.get("briefing")
        if briefing:
            reasons = {r.slot_id: r.reason for r in briefing.reasons}
            items = [i.model_copy(update={"reason": reasons.get(i.slot.slot_id, "")}) for i in items]
        return {"items": items, "final_text": render_markdown(state["request"], items, briefing,
                                                              state.get("weather", ""), state.get("errors", []))}

    g = StateGraph(PlanState)
    for name, fn in [("resolve_region", resolve_region), ("extract_preferences", extract_preferences),
                     ("fetch_weather", fetch_weather), ("search", search), ("plan_routes", plan_routes),
                     ("brief", brief), ("render", render)]:
        g.add_node(name, fn)
    g.add_edge(START, "resolve_region")
    g.add_edge("resolve_region", "extract_preferences")
    g.add_edge("resolve_region", "fetch_weather")
    g.add_edge("extract_preferences", "search")
    g.add_edge("search", "plan_routes")
    g.add_edge(["search", "fetch_weather"], "brief")       # 둘 다 끝나야 실행 (join)
    g.add_edge(["plan_routes", "brief"], "render")
    g.add_edge("render", END)
    return g.compile()


async def stream_plan(app, request: dict, on_plan=None) -> dict:
    """워크플로우를 스트리밍 실행. 시간표가 나오는 즉시(브리핑 전) on_plan(임시 마크다운)을 한 번 호출하고 최종 상태를 반환.

    UI 체감 지연을 줄이기 위한 것: 사용자는 LLM 브리핑을 기다리지 않고 일정부터 본다.
    """
    final, shown = {}, False
    async for state in app.astream({"request": request}, stream_mode="values"):
        final = state
        items = state.get("items") or []
        if on_plan and not shown and not state.get("final_text") and any(i.start for i in items):
            on_plan(render_markdown(request, items, None, state.get("weather", ""), state.get("errors", []))
                    + "\n\n_추천 사유를 작성하고 있어요..._")
            shown = True
    return final
