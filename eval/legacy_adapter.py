"""src/ 코드를 평가 러너에서 호출하기 위한 어댑터."""


def _tools():
    from src import tools
    return tools


def doc_to_cand(doc) -> dict:
    md = doc.metadata
    return {"name": md.get("장소명", ""), "region": md.get("지역", ""), "category": md.get("카테고리", "")}


async def legacy_retrieve(query, destination, category_filter, user_info="", anchor=None) -> dict:
    """src/tools.retrieve_candidates 호출. 앱은 anchor = current_anchor or destination을 주입하므로 기본값은 destination.

    baseline-v0 태그의 이 함수는 같은 단계를 복제한 구현이었고, P2에서 src에 함수가 생긴 뒤 위임으로 바뀌었다.
    """
    return await _tools().retrieve_candidates(query, destination, category_filter,
                                              anchor=destination if anchor is None else anchor,
                                              user_info=user_info)


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
