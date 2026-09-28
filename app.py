# app.py (구 pages/0_info.py)

import os
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["PYTHON_GRPC_IPV6_LOOPBACK"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from datetime import date, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
from src.retrieval.store import load_faiss_index


if "executor" not in st.session_state:
    st.session_state.executor = ThreadPoolExecutor(max_workers=1)

def warm_up_task():
    """백그라운드 작업: DB 로딩"""
    print(" 백그라운드 DB 로딩 시작 (Non-blocking)")
    try:
        load_faiss_index() 
        print("백그라운드 DB 로딩 완료")
    except Exception as e:
        print(f"로딩 실패: {e}")

if "db_loading_submitted" not in st.session_state:
    st.session_state.executor.submit(warm_up_task)
    st.session_state["db_loading_submitted"] = True


st.set_page_config(page_title="여행 정보 입력", layout="centered")
st.title("📝 AI 여행 플래너 시작하기")
st.markdown("여행 계획을 시작하기 위해 아래 정보를 입력하고 버튼을 눌러주세요.")

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
    start_location_input = st.text_input("출발지 (숙소/공항)", value=st.session_state.start_location, placeholder="예: 제주공항, 하얏트 호텔")

col_date1, col_date2 = st.columns(2)
with col_date1:
    start_date = st.date_input("출발일", value=st.session_state.start_date or date.today(), min_value=date.today())
with col_date2:
    end_date = st.date_input("귀가일", value=st.session_state.end_date or (start_date + timedelta(days=1)), min_value=start_date)

st.subheader("2. 여행 스타일")
col_style1, col_style2 = st.columns(2)
with col_style1:
    gathering_type = st.selectbox("모임 성격", ["가족", "친구", "연인", "혼자"])
with col_style2:
    travel_style = st.selectbox("선호 스타일", ["맛집 탐방", "힐링/휴양", "액티비티", "문화/역사", "자연 감상"])



st.markdown("---")
st.subheader("💡 상세 취향 (선택사항)")
st.info("구체적으로 적을수록 AI가 더 정확한 장소를 추천해 드려요!")

detail_preference = st.text_area(
    "나만의 여행 스타일을 자유롭게 적어주세요",
    placeholder="예시:\n- 해산물을 좋아하고 바다가 보이는 식당을 원해요.\n- 걷는 것을 싫어해서 동선이 짧았으면 좋겠어요.",
    height=150
)

st.markdown("---")

if st.button("AI 플래너에게 정보 전달하고 시작하기", type="primary", use_container_width=True):
    if destination_input and start_date and end_date:
        st.session_state.destination = destination_input
        st.session_state.start_location = start_location_input
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.group_type = gathering_type 
        
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