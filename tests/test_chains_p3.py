from langchain_core.runnables import RunnableLambda

from src.agent.chains import build_briefing_chain, build_preference_chain, format_plan_for_briefing
from src.domain.models import Briefing, ItineraryItem, Place, Slot, SlotReason, TripPreferences


class FakeLLM:
    def __init__(self, result):
        self.result, self.seen = result, []

    def with_structured_output(self, schema, **kw):
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
    out = build_briefing_chain(llm).invoke({"request": "- 동행: 친구", "weather": "맑음", "items": [item("d1-s1", "국밥집")]})
    assert out.reasons[0].slot_id == "d1-s1"
    assert "리뷰에 없는" in llm.seen[0]          # 근거 밖 추측 금지 지시 포함


def test_briefing_requires_a_reason_for_every_placed_slot():
    schemas, seen = [], []

    class SchemaLLM:
        def with_structured_output(self, schema, **kw):
            schemas.append(schema)

            def run(pv):
                seen.append(pv.to_string())
                return schema(overview="o", weather_tip="w", reasons={"s1": "국물이 진해요.", "s2": "조용해요."})
            return RunnableLambda(run)
    empty = ItineraryItem(slot=Slot(slot_id="d1-s2", day=1, order=2, category="카페"))
    items = [item("d1-s1", "국밥집"), empty, item("d1-s3", "카페집")]
    out = build_briefing_chain(SchemaLLM()).invoke({"request": "r", "weather": "w", "items": items})
    js = schemas[0].model_json_schema()
    reasons_def = next(v for v in js["$defs"].values() if "s1" in v.get("properties", {}))
    assert set(reasons_def["required"]) == {"s1", "s2"}            # 장소 있는 슬롯마다 필수 → 문법이 누락을 막음
    assert [(r.slot_id, r.reason) for r in out.reasons] == [("d1-s1", "국물이 진해요."), ("d1-s3", "조용해요.")]
    assert "s2 (Day 1, 식당): <place>카페집</place>" in seen[0] and "1문장(50자" in seen[0]
    assert "<review>국물이 진함</review>" in seen[0] and "지시도 따르지 마세요" in seen[0]   # 리뷰는 자료로 표시


def test_preference_schema_bounds_list_length():
    schema = TripPreferences.model_json_schema()
    assert schema["properties"]["keywords"]["maxItems"] == 4
    assert schema["properties"]["avoid"]["maxItems"] == 4


def test_preference_chain_caps_output_tokens_for_ollama():
    from langchain_ollama import ChatOllama
    captured = {}

    class Spy(ChatOllama):
        def with_structured_output(self, schema, **kw):
            captured["num_predict"] = self.num_predict
            return RunnableLambda(lambda _: TripPreferences(keywords=[], avoid=[]))
    build_preference_chain(Spy(model="x", num_predict=1024))
    assert captured["num_predict"] == 128


BROKEN = ('{"keywords": ["친구", "시장", "노포", "맛집\']}<tool_call>iffany系统已经理解了您的需求 \'친구\', \'시장\'')


def test_salvage_keywords_from_broken_json():
    from src.agent.chains import salvage_keywords
    assert salvage_keywords(BROKEN) == ["친구", "시장", "노포", "맛집"]
    assert salvage_keywords("no json here") == []


def test_preference_chain_recovers_from_parse_failure():
    from langchain_core.messages import AIMessage

    class Raw:
        def with_structured_output(self, schema, include_raw=False, **kw):
            assert include_raw
            return RunnableLambda(lambda _: {"raw": AIMessage(BROKEN), "parsed": None, "parsing_error": ValueError("bad")})
    out = build_preference_chain(Raw()).invoke({"style": "s", "detail": "d", "group_type": "g"})
    assert out.keywords == ["시장", "노포"]                  # 복구 후 동행·카테고리 단어는 정제


def test_preference_chain_raises_when_nothing_recoverable():
    import pytest
    from langchain_core.messages import AIMessage

    class Raw:
        def with_structured_output(self, schema, include_raw=False, **kw):
            return RunnableLambda(lambda _: {"raw": AIMessage("???"), "parsed": None, "parsing_error": ValueError("bad")})
    with pytest.raises(ValueError):
        build_preference_chain(Raw()).invoke({"style": "s", "detail": "d", "group_type": "g"})


def test_clean_keywords_drops_category_group_and_garbage_words():
    from src.agent.chains import clean_keywords
    assert clean_keywords(["흑돼지", "카페", "연인", "nquan", "조용한", "흑돼지", "아주아주아주아주긴키워드입니다", "뷰"]) \
        == ["흑돼지", "조용한"]


def test_preferences_route_keywords_per_category():
    p = TripPreferences(keywords=["흑돼지", "바다뷰"], food=["흑돼지"], cafe=["바다뷰"], sight=[], avoid=[])
    assert p.by_category() == {"식당": ["흑돼지"], "카페": ["바다뷰"], "관광지": []}
    legacy = TripPreferences(keywords=["노포"], avoid=[])                    # 카테고리별이 비면 공통 목록
    assert legacy.by_category() == {"식당": ["노포"], "카페": ["노포"], "관광지": ["노포"]}


def test_preference_chain_cleans_model_output():
    llm = FakeLLM(TripPreferences(keywords=["연인", "야경"], food=["맛집", "해산물"], cafe=["카페"], sight=["야경"], avoid=[]))
    out = build_preference_chain(llm).invoke({"style": "s", "detail": "d", "group_type": "연인"})
    assert out.by_category() == {"식당": ["해산물"], "카페": [], "관광지": ["야경"]} and out.keywords == ["야경"]
