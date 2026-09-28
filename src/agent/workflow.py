"""초안 일정 생성 워크플로우 (LangGraph).

    parse_request → search → (plan_routes ∥ fetch_weather) → compose

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
from src.domain.models import ItineraryItem, Slot, StepError, TripPreferences
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

    async def parse_request(state: PlanState) -> dict:
        """목적지 → 행정구역(결정적, 모호하면 Maps), 취향 키워드 추출(LLM 1회), 슬롯 생성."""
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
                errors.append(StepError.from_error("parse_request", RegionNotFoundError(req["destination"])))
        try:
            prefs = await asyncio.wait_for(pref_chain.ainvoke({"group_type": req["group_type"], "style": req["style"],
                                                               "detail": req.get("detail", "")}), llm_timeout)
        except Exception as e:
            log.warning("취향 파싱 실패 → 키워드 없이 진행: %s", e)
            errors.append(StepError.from_error("parse_request", LLMOutputError("preference", e)))
            prefs = TripPreferences(keywords=[], avoid=[])
        log.info("요청 해석: %s → %s, 키워드 %s", req["destination"], region or "?", prefs.keywords)
        return {"region": region, "preferences": prefs, "slots": build_slots(req["days"]), "errors": errors}

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
            text = await asyncio.to_thread(weather_fn, req["destination"], dates)
            if not text or text.startswith(("오류", "정보:")):
                raise RuntimeError(text or "empty")
            return {"weather": text}
        except Exception as e:
            log.warning("날씨 조회 실패: %s", e)
            return {"weather": "", "errors": [StepError.from_error("fetch_weather", ExternalServiceError("날씨", str(e)))]}

    async def compose(state: PlanState) -> dict:
        """근거 리뷰 기반 브리핑(LLM 1회) + 결정적 렌더. 브리핑 실패 시 추천 사유 없이 일정만 렌더."""
        req, items, errors = state["request"], state["items"], []
        briefing = None
        if any(i.place for i in items):
            try:
                briefing = await asyncio.wait_for(brief_chain.ainvoke({"request": request_text(req),
                                                                       "weather": state.get("weather") or "정보 없음",
                                                                       "plan": format_plan_for_briefing(items)}), llm_timeout)
                reasons = {r.slot_id: r.reason for r in briefing.reasons}
                items = [i.model_copy(update={"reason": reasons.get(i.slot.slot_id, "")}) for i in items]
            except Exception as e:
                log.warning("브리핑 실패 → 사유 없이 렌더: %s", e)
                errors.append(StepError.from_error("compose", LLMOutputError("briefing", e)))
        all_errors = [*state.get("errors", []), *errors]
        return {"items": items, "errors": errors,
                "final_text": render_markdown(req, items, briefing, state.get("weather", ""), all_errors)}

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
