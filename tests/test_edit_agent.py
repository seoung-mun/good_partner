import json

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from src.agent.edit_agent import build_edit_app, edit
from src.domain.models import ItineraryItem, Place, Slot


class ToolFake(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


def doc(name, cat):
    return Document(page_content="x", metadata={"장소명": name, "지역": "부산광역시 중구", "카테고리": cat, "리뷰": "좋음"})


class Emb:
    def embed_documents(self, texts):
        return [[0.0]] * len(texts)


class DB:
    embedding_function = Emb()

    def similarity_search_by_vector(self, v, k=4):
        return [doc("기존카페", "카페"), doc("조용한카페", "카페"), doc("새식당", "식당")]


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
