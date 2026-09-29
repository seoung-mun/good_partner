import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from src.agent.edit_agent import build_edit_app, edit
from src.domain.models import ItineraryItem, Place, Slot
from tests.fakes import make_db, meta


class ToolFake(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self




def DB():
    return make_db([meta("기존카페", "부산광역시 중구", "카페", "기존카페 좋음"),
                    meta("조용한카페", "부산광역시 중구", "카페", "조용한카페 좋음"),
                    meta("새식당", "부산광역시 중구", "식당", "새식당 좋음")])


async def route(a, b, mode="transit", departure_time=None):
    return {"duration_value": 600, "steps": []}


def plan():
    mk = lambda sid, o, cat, name: ItineraryItem(slot=Slot(slot_id=sid, day=1, order=o, category=cat),
                                                  place=Place(name=name, region="부산광역시 중구", category=cat))
    return [mk("d1-s1", 1, "식당", "국밥집"), mk("d1-s2", 2, "카페", "기존카페"), mk("d1-s3", 3, "관광지", "공원")]


STATE = {"request": {"destination": "부산 중구", "days": 1, "group_type": "친구", "style": "맛집 탐방"},
         "region": "부산광역시 중구"}


def app_with(*msgs):
    llm = ToolFake(messages=iter(list(msgs)))
    return build_edit_app(llm=llm, load_db=lambda: DB(), route_fn=route)


def call(name, args, cid="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


def tool_payloads(out):
    return [json.loads(m.content) for m in out["messages"] if isinstance(m, ToolMessage)]


async def test_remove_slot():
    out = await edit(app_with(call("remove_slot", {"slot_id": "d1-s2"}), AIMessage("삭제했어요")),
                     {**STATE, "items": plan()}, "카페 빼줘")
    assert [i.slot.slot_id for i in out["items"]] == ["d1-s1", "d1-s3"]
    assert tool_payloads(out)[0]["ok"] is True
    assert "기존카페" not in out["final_text"] and "국밥집" in out["final_text"]


async def test_replace_slot_searches_and_excludes_current():
    out = await edit(app_with(call("replace_slot", {"slot_id": "d1-s2", "query": "조용한"}), AIMessage("바꿨어요")),
                     {**STATE, "items": plan()}, "카페 조용한 데로 바꿔줘")
    assert out["items"][1].place.name == "조용한카페"
    assert out["items"][2].move_min == 10               # 변경 후 시간표 재계산


async def test_invalid_slot_returns_error_without_change():
    out = await edit(app_with(call("remove_slot", {"slot_id": "d9-s9"}), AIMessage("없는 슬롯이에요")),
                     {**STATE, "items": plan()}, "이상한 요청")
    assert [i.slot.slot_id for i in out["items"]] == ["d1-s1", "d1-s2", "d1-s3"]
    p = tool_payloads(out)[0]
    assert p["ok"] is False and "d9-s9" in p["error"]


async def test_add_slot_appends_to_day():
    out = await edit(app_with(call("add_slot", {"day": 1, "category": "식당", "query": ""}), AIMessage("추가")),
                     {**STATE, "items": plan()}, "저녁 하나 추가")
    assert out["items"][-1].slot.slot_id == "d1-s4" and out["items"][-1].place.name == "새식당"


async def test_tool_exception_becomes_error_envelope():
    def broken_db():
        raise RuntimeError("index missing")
    llm = ToolFake(messages=iter([call("replace_slot", {"slot_id": "d1-s2", "query": ""}), AIMessage("실패 안내")]))
    app = build_edit_app(llm=llm, load_db=broken_db, route_fn=route)
    out = await edit(app, {**STATE, "items": plan()}, "카페 바꿔줘")
    p = tool_payloads(out)[0]
    assert p["ok"] is False and "index missing" in p["error"]
    assert out["items"][1].place.name == "기존카페"


async def test_replaced_place_is_rejected_and_never_comes_back():
    # 기존카페 → 조용한카페로 교체 후, 다시 교체해도 방금 뺀 기존카페는 후보가 아님 (남은 카페 없음 → 실패)
    out = await edit(app_with(call("replace_slot", {"slot_id": "d1-s2", "query": ""}), AIMessage("바꿨어요")),
                     {**STATE, "items": plan()}, "카페 바꿔줘")
    assert ("기존카페", "부산광역시 중구") in out["rejected"]
    out2 = await edit(app_with(call("replace_slot", {"slot_id": "d1-s2", "query": ""}), AIMessage("없어요")),
                      {**STATE, "items": out["items"], "rejected": out["rejected"]}, "또 바꿔줘")
    assert tool_payloads(out2)[-1]["ok"] is False and out2["items"][1].place.name == "조용한카페"


async def test_removed_place_is_not_added_back():
    items = [i for i in plan() if i.slot.slot_id != "d1-s1"]      # 식당 없음
    out = await edit(app_with(call("add_slot", {"day": 1, "category": "식당", "query": ""}), AIMessage("추가")),
                     {**STATE, "items": items, "rejected": [("새식당", "부산광역시 중구")]}, "식당 추가")
    assert tool_payloads(out)[-1]["ok"] is False


def test_swap_place_replaces_and_drops_it_from_alternatives():
    from src.domain.schedule import swap_place
    items = plan()
    new = Place(name="대안카페", region="부산광역시 중구", category="카페")
    items[1] = items[1].model_copy(update={"alternatives": [new], "reason": "옛 사유"})
    out = swap_place(items, "d1-s2", new)
    assert out[1].place.name == "대안카페" and out[1].alternatives == [] and out[1].reason == ""
    assert out[0] is items[0]


def _app(db, *msgs):
    return build_edit_app(llm=ToolFake(messages=iter(list(msgs))), load_db=lambda: db, route_fn=route)


def _it(sid, day, order, cat, name, region="부산광역시 중구", meal=None):
    return ItineraryItem(slot=Slot(slot_id=sid, day=day, order=order, category=cat, meal=meal),
                         place=Place(name=name, region=region, category=cat))


async def test_replacing_breakfast_searches_breakfast_places():
    db = make_db([meta("고깃집", "부산광역시 중구", "식당", "삼겹살"), meta("해장국집", "부산광역시 중구", "식당", "해장")],
                 {"삼겹살": [0.0, 0, 0, 0], "해장": [1.0, 0, 0, 0], "아침 해장국 브런치 좋은 맛집": [1.0, 0, 0, 0]})
    items = [_it("d2-s1", 2, 1, "식당", "옛아침집", meal="아침")]
    out = await edit(_app(db, call("replace_slot", {"slot_id": "d2-s1", "query": ""}), AIMessage("바꿨어요")),
                     {**STATE, "items": items}, "둘째 날 아침 바꿔줘")
    assert out["items"][0].place.name == "해장국집"            # 일반 검색 1위(고깃집)가 아니라 아침 검색


async def test_replacement_prefers_same_district_as_neighbors():
    db = make_db([meta("중구카페", "부산광역시 중구", "카페", "a"), meta("해운대카페", "부산광역시 해운대구", "카페", "b")])
    items = [_it("d1-s1", 1, 1, "식당", "해운대식당", "부산광역시 해운대구"),
             _it("d1-s2", 1, 2, "카페", "옛카페", "부산광역시 해운대구")]
    out = await edit(_app(db, call("replace_slot", {"slot_id": "d1-s2", "query": ""}), AIMessage("바꿨어요")),
                     {**STATE, "region": "부산광역시", "items": items}, "카페 바꿔줘")
    assert out["items"][1].place.name == "해운대카페"


async def test_added_sight_goes_before_dinner():
    db = make_db([meta("명소", "부산광역시 중구", "관광지", "a")])
    items = [_it("d1-s1", 1, 1, "식당", "점심집", meal="점심"), _it("d1-s4", 1, 4, "식당", "저녁집", meal="저녁")]
    out = await edit(_app(db, call("add_slot", {"day": 1, "category": "관광지", "query": ""}), AIMessage("추가")),
                     {**STATE, "items": items}, "관광지 하나 더")
    assert [i.place.name for i in out["items"]] == ["점심집", "명소", "저녁집"]


async def test_nearby_preference_never_beats_relevance_outside_top_candidates():
    rows = [meta("해산물집", "부산광역시 금정구", "식당", "sea")]
    rows += [meta(f"금정식당{i}", "부산광역시 금정구", "식당", f"g{i}") for i in range(10)]
    rows += [meta("영도고깃집", "부산광역시 영도구", "식당", "far")]           # 같은 구지만 관련도 12위
    table = {"sea": [0.0, 0, 0, 0], "해산물 좋은 맛집": [0.0, 0, 0, 0], "좋은 맛집": [0.0, 0, 0, 0],
             **{f"g{i}": [1.0 + i * 0.1, 0, 0, 0] for i in range(10)}, "far": [9.0, 0, 0, 0]}
    items = [_it("d1-s1", 1, 1, "카페", "영도카페", "부산광역시 영도구"), _it("d1-s2", 1, 2, "식당", "옛집", "부산광역시 영도구", "저녁")]
    out = await edit(_app(make_db(rows, table), call("replace_slot", {"slot_id": "d1-s2", "query": "해산물"}),
                          AIMessage("바꿨어요")), {**STATE, "region": "부산광역시", "items": items}, "저녁 해산물집으로")
    assert out["items"][1].place.name == "해산물집"                     # 인근이라는 이유로 관련 없는 곳을 고르지 않음


async def test_edit_that_pushes_dinner_late_warns_user():
    db = make_db([meta("명소", "부산광역시 중구", "관광지", "a")])
    items = [_it("d1-s1", 1, 1, "관광지", "긴명소"), _it("d1-s2", 1, 2, "관광지", "긴명소2"),
             _it("d1-s3", 1, 3, "식당", "저녁집", meal="저녁")]
    req = {**STATE["request"], "start_time": "15:00"}
    out = await edit(_app(db, call("add_slot", {"day": 1, "category": "관광지", "query": ""}), AIMessage("추가")),
                     {**STATE, "request": req, "items": items}, "관광지 하나 더")
    assert any("저녁" in e.message for e in out["errors"])
