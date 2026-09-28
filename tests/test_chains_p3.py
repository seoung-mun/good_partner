from langchain_core.runnables import RunnableLambda

from src.agent.chains import build_briefing_chain, build_preference_chain, format_plan_for_briefing
from src.domain.models import Briefing, ItineraryItem, Place, Slot, SlotReason, TripPreferences


class FakeLLM:
    def __init__(self, result):
        self.result, self.seen = result, []

    def with_structured_output(self, schema):
        def run(prompt_value):
            self.seen.append(prompt_value.to_string())
            return self.result
        return RunnableLambda(run)


def test_preference_chain_structured():
    llm = FakeLLM(TripPreferences(keywords=["노포"], avoid=[]))
    out = build_preference_chain(llm).invoke({"style": "맛집 탐방", "detail": "노포 좋아요", "group_type": "친구"})
    assert out.keywords == ["노포"] and "노포 좋아요" in llm.seen[0]


def item(sid, name):
    return ItineraryItem(slot=Slot(slot_id=sid, day=1, order=1, category="식당"),
                         place=Place(name=name, region="부산광역시 중구", category="식당", reviews=["국물이 진함"]))


def test_format_plan_includes_slot_ids_and_reviews():
    text = format_plan_for_briefing([item("d1-s1", "국밥집"), ItineraryItem(slot=Slot(slot_id="d1-s2", day=1, order=2, category="카페"))])
    assert "d1-s1" in text and "국밥집" in text and "국물이 진함" in text
    assert "d1-s2" in text and "비어 있음" in text


def test_briefing_chain_grounded_prompt():
    b = Briefing(overview="o", weather_tip="w", reasons=[SlotReason(slot_id="d1-s1", reason="r")])
    llm = FakeLLM(b)
    out = build_briefing_chain(llm).invoke({"request": "- 동행: 친구", "weather": "맑음", "plan": "d1-s1 국밥집"})
    assert out.reasons[0].slot_id == "d1-s1"
    assert "리뷰에 없는" in llm.seen[0]          # 근거 밖 추측 금지 지시 포함


def test_briefing_prompt_limits_output_length():
    from src.agent.chains import BRIEFING_PROMPT
    text = BRIEFING_PROMPT.format(request="r", weather="w", plan="p")
    assert "60자" in text and "1문장" in text
