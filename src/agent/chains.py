"""LLM 체인: 여행 요청 → 취향 키워드, 일정 → 근거 기반 브리핑. 둘 다 Pydantic 구조화 출력."""
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from pydantic import BaseModel, Field, create_model

from src.domain.models import Briefing, ItineraryItem, SlotReason, TripPreferences

PREFERENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "여행 요청에서 리뷰 검색에 쓸 취향 키워드를 뽑으세요. 키워드는 리뷰에 실제로 나올 법한 짧은 한국어 "
               "형용사·명사 1~4개(예: 조용한, 바다뷰, 노포, 아이와). 지역명·카테고리명(맛집, 카페, 관광지)은 넣지 마세요."),
    ("human", "동행: {group_type}\n스타일: {style}\n상세 요청: {detail}"),
])

BRIEFING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "당신은 여행 일정 가이드입니다. 주어진 일정과 각 장소의 리뷰만 근거로 브리핑을 작성하세요. "
               "리뷰에 없는 사실(메뉴, 시설, 가격 등)은 추측하지 마세요. 장소를 바꾸거나 추가하지 마세요. "
               "overview는 1~2문장, weather_tip은 1문장. reasons에는 일정의 각 번호(s1, s2, …)마다 그 장소의 리뷰를 근거로 "
               "사용자 요청과 연결한 추천 사유를 1~2문장으로 쓰세요."),
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




def build_preference_chain(llm):
    """여행 요청 → TripPreferences(취향 키워드) 체인. 출력 토큰 상한을 이 체인에만 더 낮게 걸고, 깨진 출력은 복구."""
    if getattr(llm, "num_predict", None):
        llm = llm.model_copy(update={"num_predict": PREFERENCE_MAX_TOKENS})
    structured = llm.with_structured_output(TripPreferences, include_raw=True)
    return PREFERENCE_PROMPT | structured | RunnableLambda(_preferences_or_salvage)


def _plan_with_keys(items: list[ItineraryItem]) -> tuple[str, dict[str, str]]:
    """브리핑 입력 일정: 장소 있는 슬롯에 짧은 번호(s1, s2…)를 붙인다 → (일정 텍스트, 번호 → slot_id)."""
    lines, keys = [], {}
    for it in items:
        if it.place is None:
            continue
        key = f"s{len(keys) + 1}"
        keys[key] = it.slot.slot_id
        reviews = " / ".join(r[:120] for r in it.place.reviews[:2])
        lines.append(f"- {key} (Day {it.slot.day}, {it.slot.meal or it.slot.category}): {it.place.name} 리뷰: {reviews}")
    return "\n".join(lines), keys


def _briefing_schema(items: list[ItineraryItem], keys: dict[str, str]) -> type[BaseModel]:
    """이번 일정 전용 출력 스키마: 장소마다 필수 사유 필드 → 구조화 출력 문법이 사유 누락을 막는다."""
    names = {it.slot.slot_id: it.place.name for it in items if it.place}
    reasons = create_model("SlotReasons", **{k: (str, Field(description=f"{names[sid]} 추천 사유 1~2문장"))
                                             for k, sid in keys.items()})
    return create_model("PlanBriefing", overview=(str, Field(description="일정 전체를 소개하는 1~2문장")),
                        weather_tip=(str, Field(description="날씨 데이터를 한 문장으로 요약한 조언")),
                        reasons=(reasons, ...))


def build_briefing_chain(llm):
    """일정 + 근거 리뷰 → Briefing(개요·날씨 팁·슬롯별 사유). 입력: {request, weather, items}.

    호출마다 일정에 맞춘 스키마(장소마다 필수 사유)를 만들어 쓰므로 소형 모델도 사유를 빠뜨리지 않는다.
    """
    async def run(x: dict) -> Briefing:
        plan, keys = _plan_with_keys(x["items"])
        schema = _briefing_schema(x["items"], keys)
        out = await (BRIEFING_PROMPT | llm.with_structured_output(schema)).ainvoke(
            {"request": x["request"], "weather": x["weather"], "plan": plan})
        if isinstance(out, Briefing):   # 이미 Briefing을 돌려주는 모델(테스트 대역 등)
            return out
        got = out.reasons.model_dump()
        return Briefing(overview=out.overview, weather_tip=out.weather_tip,
                        reasons=[SlotReason(slot_id=sid, reason=got[k]) for k, sid in keys.items() if got.get(k)])

    def run_sync(x: dict) -> Briefing:
        import asyncio
        return asyncio.run(run(x))
    return RunnableLambda(run_sync, afunc=run)


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
