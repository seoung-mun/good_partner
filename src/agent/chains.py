"""LLM 체인: 여행 요청 → 취향 키워드, 일정 → 근거 기반 브리핑. 둘 다 Pydantic 구조화 출력."""
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

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


PREFERENCE_MAX_TOKENS = 128   # 정상 출력은 수십 토큰. 폭주 생성(관측: 1,300토큰·45초)을 짧게 끊는다


def salvage_keywords(text: str, limit: int = 4) -> list[str]:
    """깨진 구조화 출력에서 keywords 목록만 복구. 소형 모델이 따옴표를 잘못 닫고 잡담을 이어 붙이는 경우 대비."""
    m = re.search(r'"keywords"\s*:\s*\[(.*?)(?:\]|<|\n|$)', text, re.S)
    if not m:
        return []
    return re.findall(r'["\']([^"\'\[\],]{1,15})["\']', m.group(1))[:limit]


def _preferences_or_salvage(result) -> TripPreferences:
    """include_raw 결과 → 파싱 성공이면 그대로, 실패면 원문에서 키워드 복구, 그래도 없으면 예외(호출부가 StepError 기록)."""
    if isinstance(result, TripPreferences):
        return result
    if result.get("parsed") is not None:
        return result["parsed"]
    raw = getattr(result.get("raw"), "content", "") or ""
    keywords = salvage_keywords(raw if isinstance(raw, str) else str(raw))
    if not keywords:
        raise ValueError(f"취향 파싱 실패: {result.get('parsing_error')}")
    return TripPreferences(keywords=keywords, avoid=[])


PREFERENCE_MAX_TOKENS = 128   # 정상 출력은 수십 토큰. 폭주 생성(관측: 1,300토큰·45초)을 짧게 끊는다


def build_preference_chain(llm):
    """여행 요청 → TripPreferences(취향 키워드) 체인. 출력 토큰 상한을 이 체인에만 더 낮게 걸고, 깨진 출력은 복구."""
    if getattr(llm, "num_predict", None):
        llm = llm.model_copy(update={"num_predict": PREFERENCE_MAX_TOKENS})
    structured = llm.with_structured_output(TripPreferences, include_raw=True)
    return PREFERENCE_PROMPT | structured | RunnableLambda(_preferences_or_salvage)


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
