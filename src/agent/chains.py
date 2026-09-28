"""LLM 체인: 여행 요청 → 취향 키워드, 일정 → 근거 기반 브리핑. 둘 다 Pydantic 구조화 출력."""
from langchain_core.prompts import ChatPromptTemplate

from src.domain.models import Briefing, ItineraryItem, TripPreferences

PREFERENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "여행 요청에서 리뷰 검색에 쓸 취향 키워드를 뽑으세요. 키워드는 리뷰에 실제로 나올 법한 짧은 한국어 "
               "형용사·명사 1~4개(예: 조용한, 바다뷰, 노포, 아이와). 지역명·카테고리명(맛집, 카페, 관광지)은 넣지 마세요."),
    ("human", "동행: {group_type}\n스타일: {style}\n상세 요청: {detail}"),
])

BRIEFING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "당신은 여행 일정 가이드입니다. 주어진 일정과 각 장소의 리뷰만 근거로 브리핑을 작성하세요. "
               "리뷰에 없는 사실(메뉴, 시설, 가격 등)은 추측하지 마세요. 장소를 바꾸거나 추가하지 마세요. "
               "overview는 1~2문장, weather_tip은 1문장. reasons에는 장소가 있는 slot_id마다 사용자 요청과 연결한 "
               "추천 사유를 1문장(60자 이내)으로 쓰세요."),
    ("human", "[사용자 요청]\n{request}\n\n[날씨]\n{weather}\n\n[일정]\n{plan}"),
])


def build_preference_chain(llm):
    """여행 요청 → TripPreferences(취향 키워드) 체인."""
    return PREFERENCE_PROMPT | llm.with_structured_output(TripPreferences)


def build_briefing_chain(llm):
    """일정 + 근거 리뷰 → Briefing(개요·날씨 팁·슬롯별 사유) 체인."""
    return BRIEFING_PROMPT | llm.with_structured_output(Briefing)


def format_plan_for_briefing(items: list[ItineraryItem]) -> str:
    """브리핑·수정 에이전트 입력용 일정 텍스트 (slot_id, 장소, 근거 리뷰 2개)."""
    lines = []
    for it in items:
        head = f"- {it.slot.slot_id} (Day {it.slot.day}, {it.slot.meal or it.slot.category})"
        if it.place is None:
            lines.append(f"{head}: 비어 있음")
            continue
        reviews = " / ".join(r[:120] for r in it.place.reviews[:2])
        lines.append(f"{head}: {it.place.name} [{it.place.region}] 리뷰: {reviews}")
    return "\n".join(lines)
