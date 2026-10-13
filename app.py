# app.py (구 pages/0_info.py)

import os
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["PYTHON_GRPC_IPV6_LOOPBACK"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# macOS: torch·faiss가 각자 libomp를 로드 → 중복 허용 + 단일 스레드로 세그폴트·교착 방지
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import streamlit as st
from datetime import date, time, timedelta
from concurrent.futures import ThreadPoolExecutor
from src.domain.schedule import check_trip_times
from src.retrieval.store import load_faiss_index
from src.logging_config import get_logger

log = get_logger(__name__)


if "executor" not in st.session_state:
    st.session_state.executor = ThreadPoolExecutor(max_workers=1)

def warm_up_task():
    """백그라운드 작업: DB 로딩"""
    log.info(" 백그라운드 DB 로딩 시작 (Non-blocking)")
    try:
        load_faiss_index() 
        log.info("백그라운드 DB 로딩 완료")
    except Exception:
        log.exception("백그라운드 DB 로딩 실패 (일정 페이지에서 다시 시도하며 오류를 표시함)")

if "db_loading_submitted" not in st.session_state:
    st.session_state.executor.submit(warm_up_task)
    st.session_state["db_loading_submitted"] = True


st.set_page_config(page_title="여행 정보 입력", layout="centered")
st.title("AI 여행 플래너")
st.caption("실제 방문자 리뷰를 근거로, 동선까지 고려한 일정을 만들어 드려요.")

defaults = {
    "destination": "", "start_location": "", "start_date": None, "end_date": None,
    "user_preferences": {}, "activity_level": 3, "preferences_collected": False,
    "dates": "", "preference": "", "total_days": 1, "current_planning_day": 1,
    "itinerary": [], "messages": []
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

st.subheader("1. 기본 정보")

col_dest, col_start = st.columns(2)
with col_dest:
    destination_input = st.text_input("목적지", value=st.session_state.destination, placeholder="예: 부산, 제주도")
with col_start:
    start_location_input = st.text_input("숙소 (또는 출발지)", value=st.session_state.start_location, placeholder="예: 부산역, 해운대 숙소 (매일 여기서 출발·복귀)")

col_date1, col_date2 = st.columns(2)
with col_date1:
    start_date = st.date_input("출발일", value=st.session_state.start_date or date.today(), min_value=date.today())
with col_date2:
    end_date = st.date_input("귀가일", value=st.session_state.end_date or (start_date + timedelta(days=1)), min_value=start_date)

col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    travel_mode = st.radio("이동 수단", ["대중교통", "자동차"], horizontal=True)
with col_t2:
    arrive_time = st.time_input("첫날 도착 시각", value=time(12, 0), step=1800)
with col_t3:
    leave_time = st.time_input("마지막 날 출발 시각", value=time(20, 0), step=1800)

st.subheader("2. 여행 스타일")
col_style1, col_style2 = st.columns(2)
with col_style1:
    gathering_type = st.selectbox("모임 성격", ["가족", "친구", "연인", "혼자"])
with col_style2:
    travel_style = st.selectbox("선호 스타일", ["맛집 탐방", "힐링/휴양", "액티비티", "문화/역사", "자연 감상"])



st.markdown("---")
st.subheader("상세 취향 (선택)")
st.caption("구체적으로 적을수록 리뷰에서 더 잘 맞는 장소를 찾아요.")

detail_preference = st.text_area(
    "나만의 여행 스타일을 자유롭게 적어주세요",
    placeholder="예시:\n- 해산물을 좋아하고 바다가 보이는 식당을 원해요.\n- 걷는 것을 싫어해서 동선이 짧았으면 좋겠어요.",
    height=150
)

st.markdown("---")

time_error = check_trip_times((end_date - start_date).days + 1, arrive_time.strftime("%H:%M"), leave_time.strftime("%H:%M"))
if time_error:
    st.warning(time_error)

if st.button("일정 만들기", type="primary", use_container_width=True):
    if time_error:
        st.error(time_error)
    elif destination_input and start_date and end_date:
        st.session_state.destination = destination_input
        st.session_state.start_location = start_location_input
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.group_type = gathering_type 
        st.session_state.travel_style = travel_style
        st.session_state.detail_preference = detail_preference
        st.session_state.travel_mode = "driving" if travel_mode == "자동차" else "transit"
        st.session_state.arrive_time = arrive_time.strftime("%H:%M")
        st.session_state.leave_time = leave_time.strftime("%H:%M")
        st.session_state.plan = None
        
        days = (end_date - start_date).days + 1 
        travel_dates_str = f"{start_date.strftime('%Y년 %m월 %d일')}부터 {days}일간"
        st.session_state.dates = travel_dates_str
        st.session_state.total_days = days

        pref_list = [
            f"- 동행: {gathering_type}",
            f"- 스타일: {travel_style}"
        ]
        if start_location_input:
            pref_list.append(f"- 출발지: {start_location_input}")
        if detail_preference.strip():
            pref_list.append(f"- 상세 요청: {detail_preference}")
            
        st.session_state.preference = "\n".join(pref_list)

        st.session_state.preferences_collected = True
        st.session_state.messages = []
        st.session_state.itinerary = []
        
        st.switch_page("pages/trip_planner.py") 

    else:
        st.error("목적지와 날짜는 반드시 입력해야 합니다.")