"""기존 src/ 코드를 수정 없이 호출하는 어댑터. baseline-v0 태그 시점의 src에서만 유효."""
import asyncio


def _tools():
    from src import tools
    return tools


def doc_to_cand(doc) -> dict:
    md = doc.metadata
    return {"name": md.get("장소명", ""), "region": md.get("지역", ""), "category": md.get("카테고리", "")}


async def legacy_retrieve(query, destination, category_filter, user_info="", anchor=None) -> dict:
    """src/tools.py find_and_select_best_place의 검색 단계 재현 (추천사 생성·최종 1곳 선택 제외)."""
    T = _tools()
    anchor = anchor if anchor is not None else destination
    target_input = anchor if anchor else query
    target_region = ""
    try:
        target_region = (await T.region_chain.ainvoke({"query": target_input, "destination": destination})).strip()
    except Exception as e:
        print(f"region_chain 실패: {e}")
    if not target_region or target_region == destination:
        if anchor:
            target_region = await T.resolve_admin_region(anchor, destination)
        else:
            t_inp = query if destination in query else f"{destination} {query}"
            target_region = await T.resolve_admin_region(t_inp, destination)
    target_region = target_region.strip()

    center_lat = center_lng = None
    center_place = anchor if anchor else target_region
    if center_place:
        center_lat, center_lng = await T.get_coordinates(center_place)

    try:
        out = await T.query_gen_chain.ainvoke({"target_region": target_region, "query": query,
                                                "user_info": user_info, "category_filter": category_filter})
        search_queries = [q.strip() for q in out.split(",") if q.strip()]
    except Exception:
        search_queries = [f"{target_region} {query} {category_filter}"]

    results = await asyncio.gather(*[T._search_docs(q, k=50) for q in search_queries])
    seen, aggregated = set(), []
    for docs in results:
        for d in docs:
            name = d.metadata.get("장소명", "")
            if name and name not in seen:
                seen.add(name)
                aggregated.append(d)
    candidates = await T._filter_candidates(aggregated, target_region, [], category_filter)

    fallback_used = False
    if not candidates:
        fallback_used = True
        docs_v2 = await T._search_docs(f"{target_region} {query} {category_filter}", k=30)
        candidates = await T._filter_candidates(docs_v2, target_region, [], category_filter)
        if candidates and center_lat and center_lng:
            scored = []
            for d in candidates[:5]:
                md = d.metadata
                addr = md.get("지역", "") or md.get("road_address", "") or md.get("상세 주소", "")
                lat, lng = await T.get_coordinates(addr)
                dist = T.calculate_haversine_distance(center_lat, center_lng, lat, lng) if lat and lng else 9999.0
                scored.append((dist, d))
            scored.sort(key=lambda x: x[0])
            candidates = [d for _, d in scored]
    return {"target_region": target_region, "queries": search_queries,
            "candidates": candidates, "fallback_used": fallback_used}


from datetime import date


def build_preference(trip: dict) -> str:
    """app.py의 pref_list 조립 재현."""
    lines = [f"- 동행: {trip['group_type']}", f"- 스타일: {trip['style']}"]
    if trip.get("start_location"):
        lines.append(f"- 출발지: {trip['start_location']}")
    if trip.get("detail", "").strip():
        lines.append(f"- 상세 요청: {trip['detail']}")
    return "\n".join(lines)


def initial_state(trip: dict, start: date) -> dict:
    """pages/trip_planner.py run_ai_agent()의 current_state + 초기 프롬프트 재현."""
    from langchain_core.messages import HumanMessage
    days = trip["days"]
    dates = f"{start.strftime('%Y년 %m월 %d일')}부터 {days}일간"
    pref = build_preference(trip)
    prompt = f"""
    안녕하세요! 아래 정보로 여행 계획을 세워주세요.
    - 목적지: {trip['destination']}
    - 일정: {dates} (총 {days}일)
    - 스타일: {pref}
    - 동행: {trip['group_type']}
    
    날씨 확인 후, 1일차 일정부터 바로 시작해주세요.
    """
    return {"messages": [HumanMessage(content=prompt)], "itinerary": [],
            "destination": trip["destination"], "dates": dates, "group_type": trip["group_type"],
            "style": pref, "preference": pref, "total_days": days, "current_weather": "",
            "show_pdf_button": False, "current_anchor": trip["destination"],
            "dialog_stage": "planning", "last_deleted_spot": None, "ban_list": []}


async def run_legacy_trip(app, trip: dict, start: date) -> dict:
    from langchain_core.messages import AIMessage
    config = {"configurable": {"thread_id": f"eval-{trip['id']}"}, "recursion_limit": 50}
    try:
        res = await app.ainvoke(initial_state(trip, start), config=config)
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "detail": str(e)[:300],
                "itinerary": [], "final_message": ""}
    ai = [m for m in res.get("messages", []) if isinstance(m, AIMessage) and m.content]
    return {"ok": True, "error": None, "itinerary": res.get("itinerary", []),
            "final_message": ai[-1].content if ai else ""}
