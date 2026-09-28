import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.edit_agent import build_edit_app, edit
from src.agent.workflow import build_workflow, reschedule, stream_plan, to_legacy_itinerary
from src.domain.schedule import swap_place
from src.retrieval.catalog import get_catalog
from src.services.maps import get_detailed_route
from src.ui.cards import chips, move_line, place_card
from src.ui.map import day_map
from src.ui.text import to_plain
import re
import asyncio
from datetime import date, datetime
from fpdf import FPDF
import time
import os
from fpdf.enums import XPos, YPos
from src.retrieval.store import load_faiss_index
from src.logging_config import get_logger

log = get_logger(__name__)

with st.spinner("여행 데이터를 불러오는 중입니다..."):
    DB = load_faiss_index()
    log.info("1_trip_planner 페이지에서 DB 로드 확인 완료")

def normalize_to_string(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.append(str(item.get('text', '')))
            else:
                texts.append(str(item))
        return "\n".join(texts)
    return str(content)

def translate_category_to_korean(category):
    """카테고리를 한글로 변환"""
    if not category:
        return "활동"

    category_lower = str(category).lower()

    if 'activity' in category_lower:
        return "활동"
    elif 'restaurant' in category_lower or 'food' in category_lower:
        return "식당"
    elif 'cafe' in category_lower or 'coffee' in category_lower:
        return "카페"
    elif 'tourist' in category_lower or 'attraction' in category_lower:
        return "관광지"
    elif 'move' in category_lower or 'transport' in category_lower:
        return "이동"

    return category

def _normalize_itinerary_for_pdf(itinerary, total_days=None):
    norm = []
    for item in itinerary:
        if not isinstance(item, dict):
            continue
        it = item.copy()
        
        day = it.get('day', 1)
        if isinstance(day, str):
            m = re.search(r'(\d+)', day)
            try:
                day = int(m.group(1)) if m else 1
            except:
                day = 1
        else:
            try: day = int(day)
            except: day = 1
            
        if total_days:
            try:
                td = int(total_days)
                if day < 1: day = 1
                if day > td: day = td
            except: pass
            
        it['day'] = day
        if 'description' not in it: it['description'] = it.get('description', '')
        if 'type' not in it and 'category' in it: it['type'] = it.get('category')
        if 'name' not in it: it['name'] = it.get('장소명', it.get('name', '이름 없음'))
        if 'reviews' not in it: it['reviews'] = [] 
        
        norm.append(it)
    return norm

def create_itinerary_pdf(itinerary, destination, dates, weather, final_routes, total_days, start_location=None):
    pdf = FPDF()
    pdf.add_page()

    font_path = 'NanumGothic.ttf'
    bold_font_path = 'NanumGothicBold.ttf'

    has_korean_font = False
    try:
        if os.path.exists(font_path):
            pdf.add_font('NanumGothic', '', font_path)
            if os.path.exists(bold_font_path):
                pdf.add_font('NanumGothic', 'B', bold_font_path)
            else:
                pdf.add_font('NanumGothic', 'B', font_path)

            pdf.set_font('NanumGothic', '', 12)
            has_korean_font = True
        else:
            pdf.set_font('Arial', '', 12)
    except Exception as e:
        log.warning(f" [PDF 생성] 폰트 로드 에러: {e}")
        pdf.set_font('Arial', '', 12)

    pdf.set_font_size(24)
    pdf.cell(0, 20, text=f"{destination} 여행 계획", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    pdf.set_font_size(12)
    pdf.cell(0, 10, text=f"기간: {dates}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    if start_location:
        pdf.set_font_size(11)
        pdf.cell(0, 8, text=f"출발지/숙소: {start_location}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    if weather and weather.strip() and weather != '정보 없음':
        pdf.set_font_size(10)
        pdf.multi_cell(0, 5, text=f"날씨: {to_plain(weather)}", align='C')

    pdf.ln(10)

    normalized_itinerary = _normalize_itinerary_for_pdf(itinerary, total_days)
    try:
        sorted_itinerary = sorted(enumerate(normalized_itinerary), key=lambda x: (int(x[1].get('day', 1)), x[1].get('start', '00:00') or '00:00', x[0]))
        sorted_itinerary = [item[1] for item in sorted_itinerary]
    except:
        sorted_itinerary = normalized_itinerary

    for day_num in range(1, total_days + 1):
        if day_num > 1: pdf.ln(15) 

        pdf.set_font_size(18)
        if has_korean_font: pdf.set_font('NanumGothic', 'B', 18)
        pdf.cell(0, 15, text=f"Day {day_num}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font_size(11)
        if has_korean_font: pdf.set_font('NanumGothic', '', 11)

        items_today = [item for item in sorted_itinerary if int(item.get('day', 1)) == day_num]

        if not items_today:
            pdf.cell(0, 10, text="  - 계획된 일정이 없습니다.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(10)
            continue

        for item in items_today:
            item_type = item.get('type', 'activity')

            if item_type == 'move':
                pdf.set_text_color(100, 100, 100)
                pdf.set_font_size(10)
                move_text = f"      |  {item.get('duration_text', '')} : {to_plain(item.get('transport', '이동'))}"
                pdf.cell(0, 8, text=move_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_text_color(0, 0, 0)
                pdf.set_font_size(11)

            else:
                time_info = f"[{item.get('start', '시간 미정')}-{item.get('end', '')}]" if item.get('start') else "[시간 미정]"

                if has_korean_font: pdf.set_font('NanumGothic', 'B', 12)
                category_text = translate_category_to_korean(item.get('category', item_type))
                main_text = f"  ● {time_info} {to_plain(item.get('name', '이름 없음'))} ({category_text})"
                pdf.cell(0, 8, text=main_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

                if item.get('description'):
                    if has_korean_font: pdf.set_font('NanumGothic', '', 10)
                    pdf.set_x(20)
                    pdf.multi_cell(0, 5, text=to_plain(item['description']), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.ln(2)

                reviews = item.get('reviews', [])
                if reviews and isinstance(reviews, list):
                    if has_korean_font: pdf.set_font('NanumGothic', '', 9)
                    for review in reviews:
                        if isinstance(review, str): review_text = review
                        elif isinstance(review, dict): review_text = review.get('text', str(review))
                        else: review_text = str(review)
                        # 매 줄 왼쪽 여백에서 시작 — 기본값(new_x=RIGHT)이면 두 번째 리뷰부터 쓸 공간이 없어 예외
                        pdf.set_x(20)
                        pdf.multi_cell(0, 4, text=f"  • {to_plain(review_text)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.ln(2)

        pdf.ln(10)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(5)
        pdf.set_font_size(14)
        if has_korean_font: pdf.set_font('NanumGothic', '', 14)
        pdf.cell(0, 10, text="메모:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(20)

    return bytes(pdf.output())

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


async def create_plan(status):
    """초안 생성. 노드가 끝날 때마다 진행 단계를 표시."""
    out = await stream_plan(WORKFLOW, build_request(),
                            on_step=lambda n: status.write(f":material/check: {STEP_LABEL.get(n, n)}"))
    st.session_state.plan = {**{k: out.get(k) for k in ("request", "region", "items", "weather", "briefing", "errors")},
                             "keywords": out["preferences"].keywords if out.get("preferences") else [],
                             "rejected": []}


async def edit_plan(text: str):
    plan = st.session_state.plan
    out = await edit(EDIT_APP, {k: plan[k] for k in ("request", "region", "items", "rejected")}, text)
    st.session_state.plan = {**plan, "items": out["items"], "rejected": out.get("rejected") or plan["rejected"]}
    reply = next((m.content for m in reversed(out["messages"]) if isinstance(m, AIMessage) and m.content), "")
    st.session_state.messages.append(AIMessage(content=reply or "일정을 수정했어요."))


def swap(slot_id: str, place):
    """카드의 대안 선택 → 교체, 뺀 장소는 재추천 금지, 시간표 재계산."""
    plan = st.session_state.plan
    old = next(i.place for i in plan["items"] if i.slot.slot_id == slot_id)
    items, _ = loop.run_until_complete(reschedule(swap_place(plan["items"], slot_id, place), get_detailed_route,
                                                  plan["request"]))
    st.session_state.plan = {**plan, "items": items, "rejected": [*plan["rejected"], (old.name, old.region)]}


# --- 초기 실행: 초안 생성 ---
if st.session_state.plan is None:
    with st.status("여행 계획을 만드는 중이에요", expanded=True) as status:
        loop.run_until_complete(create_plan(status))
        status.update(label="여행 계획을 만들었어요", state="complete", expanded=False)
    st.rerun()

plan = st.session_state.plan
req, items, briefing = plan["request"], plan["items"], plan.get("briefing")
catalog = get_catalog(DB) if DB is not None else None

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
for day, tab in zip(days, st.tabs([f"Day {d}" for d in days])):
    today = [i for i in items if i.slot.day == day]
    with tab:
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            n = 0
            for it in today:
                if it.place is None:
                    st.caption(f"{it.slot.meal or it.slot.category}: 조건에 맞는 장소를 찾지 못했어요")
                    continue
                n += 1
                if it.move_min is not None:
                    move_line(it.move_min, it.move_detail)
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
    pdf_bytes = create_itinerary_pdf(to_legacy_itinerary(items), req["destination"], st.session_state.get("dates", ""),
                                     plan.get("weather") or "날씨 정보 없음", "", req["days"], req.get("start_location"))
    if pdf_bytes:
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
