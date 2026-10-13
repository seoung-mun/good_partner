import asyncio
from datetime import date

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.edit_agent import build_edit_app, edit
from src.agent.workflow import build_workflow, reschedule, stream_plan
from src.domain.schedule import swap_place
from src.logging_config import get_logger
from src.retrieval.catalog import get_catalog
from src.retrieval.store import load_faiss_index
from src.services.maps import get_detailed_route
from src.ui.cards import chips, move_line, place_card
from src.ui.map import day_map
from src.ui.pdf import create_itinerary_pdf

log = get_logger(__name__)

with st.spinner("여행 데이터를 불러오는 중입니다..."):
    try:
        DB = load_faiss_index()
    except Exception as e:
        log.exception("FAISS 인덱스 로드 실패")
        st.error(f"장소 데이터베이스를 불러오지 못했어요: {type(e).__name__}: {e}")
        st.stop()


def normalize_to_string(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(i.get("text", "")) if isinstance(i, dict) else str(i) for i in content)
    return str(content)


st.set_page_config(page_title="AI 여행 플래너", layout="wide")

if "preferences_collected" not in st.session_state:
    st.warning("정보 입력 페이지에서 먼저 여행 정보를 입력해주세요.")
    st.page_link("app.py", label="여행 정보 입력하기", icon=":material/edit:")
    st.stop()

if "messages" not in st.session_state: st.session_state.messages = []
if "plan" not in st.session_state: st.session_state.plan = None

if "event_loop" not in st.session_state:
    st.session_state.event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.event_loop)
loop = st.session_state.event_loop


@st.cache_resource(show_spinner=False)
def get_apps():
    """초안 생성은 결정적 워크플로우, 수정은 tool calling 에이전트."""
    return build_workflow(), build_edit_app()


WORKFLOW, EDIT_APP = get_apps()
STEP_LABEL = {"resolve_region": "여행 지역 확인", "extract_preferences": "취향 키워드 추출", "fetch_weather": "날씨 조회",
              "search": "리뷰에서 장소 검색", "locate": "동선 짜기", "plan_routes": "이동 경로 계산",
              "brief": "추천 이유 작성", "render": "마무리"}


def build_request() -> dict:
    start = st.session_state.get("start_date") or date.today()
    return {"destination": st.session_state.destination,
            "start_location": st.session_state.get("start_location", ""),
            "start_date": start.isoformat(), "days": int(st.session_state.get("total_days", 1)),
            "group_type": st.session_state.get("group_type", "친구"),
            "style": st.session_state.get("travel_style") or st.session_state.get("preference", ""),
            "detail": st.session_state.get("detail_preference", ""),
            "mode": st.session_state.get("travel_mode", "transit"),
            "start_time": st.session_state.get("arrive_time", "12:00"),
            "end_time": st.session_state.get("leave_time", "20:00")}


async def create_plan(status, preview):
    """초안 생성. 노드가 끝날 때마다 진행 단계를, 시간표가 나오면 브리핑(LLM)을 기다리지 않고 일정부터 표시."""
    out = await stream_plan(WORKFLOW, build_request(), on_plan=preview.markdown,
                            on_step=lambda n: status.write(f":material/check: {STEP_LABEL.get(n, n)}"))
    st.session_state.plan = {**{k: out.get(k) for k in ("request", "region", "items", "weather", "briefing", "errors")},
                             "keywords": out["preferences"].keywords if out.get("preferences") else [],
                             "rejected": []}


async def edit_plan(text: str):
    plan = st.session_state.plan
    out = await edit(EDIT_APP, {k: plan[k] for k in ("request", "region", "items", "rejected")}, text)
    st.session_state.plan = {**plan, "items": out["items"], "rejected": out.get("rejected") or plan["rejected"],
                             "errors": out.get("errors") or []}
    reply = next((m.content for m in reversed(out["messages"]) if isinstance(m, AIMessage) and m.content), "")
    st.session_state.messages.append(AIMessage(content=reply or "일정을 수정했어요."))


def swap(slot_id: str, place):
    """카드의 대안 선택 → 교체, 뺀 장소는 재추천 금지, 시간표 재계산."""
    plan = st.session_state.plan
    old = next(i.place for i in plan["items"] if i.slot.slot_id == slot_id)
    items, errors = loop.run_until_complete(reschedule(swap_place(plan["items"], slot_id, place), get_detailed_route,
                                                       plan["request"], trim=False))
    st.session_state.plan = {**plan, "items": items, "errors": errors,
                             "rejected": [*plan["rejected"], (old.name, old.region)]}


# --- 초기 실행: 초안 생성 ---
if st.session_state.plan is None:
    with st.status("여행 계획을 만드는 중이에요", expanded=True) as status:
        preview = st.empty()
        loop.run_until_complete(create_plan(status, preview))
        status.update(label="여행 계획을 만들었어요", state="complete", expanded=False)
    st.rerun()

plan = st.session_state.plan
req, items, briefing = plan["request"], plan["items"], plan.get("briefing")
catalog = get_catalog(DB)

st.title(f"{req['destination']} {req['days']}일 여행")
chips([req["group_type"], req["style"], "자동차" if req.get("mode") == "driving" else "대중교통",
       f"숙소 {req['start_location']}" if req.get("start_location") else ""], plan.get("keywords") or [])
for msg in dict.fromkeys(e.message for e in plan.get("errors") or []):
    st.warning(msg)
if briefing:
    st.markdown(briefing.overview)
    st.caption(f":material/partly_cloudy_day: {briefing.weather_tip}")
elif plan.get("weather"):
    st.caption(plan["weather"].splitlines()[0])

taken = {(i.place.name, i.place.region) for i in items if i.place} | {tuple(k) for k in plan["rejected"]}
days = sorted({i.slot.day for i in items})
# 탭 대신 선택한 날만 렌더 — 숨겨진 탭 안의 pydeck 지도는 크기 0으로 초기화돼 빈 화면이 된다
day = st.segmented_control("일차", days, format_func=lambda d: f"Day {d}", default=days[0] if days else None,
                           label_visibility="collapsed") or (days[0] if days else None)
for day in [day] if day else []:
    today = [i for i in items if i.slot.day == day]
    with st.container():
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            n = 0
            for it in today:
                if it.place is None:
                    st.caption(f"{it.slot.meal or it.slot.category}: 조건에 맞는 장소를 찾지 못했어요")
                    continue
                n += 1
                if it.move_min is not None:
                    where = ("도착지에서 · " if day == 1 else "숙소에서 · ") if it.from_lodging else ""
                    move_line(it.move_min, it.move_detail, label=where)
                reviews = catalog.places.get((it.place.name, it.place.region)).reviews if catalog else it.place.reviews
                picked = place_card(n, it, plan.get("keywords") or [], reviews, taken)
                if picked:
                    swap(it.slot.slot_id, picked)
                    st.rerun()
                if it.lodging_min is not None:
                    move_line(it.lodging_min, it.lodging_detail, label="숙소로 · ")
        with right:
            deck = day_map(today)
            if deck:
                st.pydeck_chart(deck, height=520)
            else:
                st.caption("좌표를 찾지 못해 지도를 표시할 수 없어요")

# --- PDF ---
if items:
    pdf_bytes = create_itinerary_pdf(items, req["destination"], st.session_state.get("dates", ""),
                                     plan.get("weather") or "날씨 정보 없음", req["days"], req.get("start_location"))
    st.download_button(label="여행 계획 PDF 다운로드", data=pdf_bytes, icon=":material/download:",
                       file_name=f"{req['destination']}_여행계획.pdf", mime="application/pdf")

# --- 수정 요청: 에이전트 ---
st.divider()
for msg in st.session_state.messages:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").markdown(normalize_to_string(msg.content))
if user_input := st.chat_input("수정할 내용을 입력하세요 (예: 1일차 카페를 조용한 곳으로 바꿔줘)"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.spinner("일정을 수정하는 중이에요..."):
        loop.run_until_complete(edit_plan(user_input))
    st.rerun()
