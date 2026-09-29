"""수정 에이전트 tool. 얇은 래퍼: 입력 검증(args_schema) → 도메인·검색 함수 → 상태 갱신(Command) + 결과 envelope."""
import json
from typing import Literal

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.domain.models import ItineraryItem, Slot
from src.domain.schedule import BREAKFAST_KEYWORDS, swap_place
from src.retrieval.retriever import search_candidates


class ReplaceSlotInput(BaseModel):
    """replace_slot 입력."""
    slot_id: str = Field(description="바꿀 일정 슬롯 ID (예: d1-s2)")
    query: str = Field(description="새 장소 조건 키워드 (예: 조용한, 바다뷰). 조건이 없으면 빈 문자열")


class RemoveSlotInput(BaseModel):
    """remove_slot 입력."""
    slot_id: str = Field(description="삭제할 일정 슬롯 ID (예: d1-s2)")


class AddSlotInput(BaseModel):
    """add_slot 입력."""
    day: int = Field(description="추가할 일차 (1부터)")
    category: Literal["식당", "카페", "관광지"] = Field(description="추가할 장소 종류")
    query: str = Field(description="장소 조건 키워드. 없으면 빈 문자열")


def _result(runtime: ToolRuntime, name: str, ok: bool, items=None, dropped=None, **payload) -> Command:
    """tool 결과를 envelope(ToolMessage)로 만들고, 성공 시 일정(items)과 뺀 장소 목록(rejected)도 갱신하는 Command."""
    msg = ToolMessage(content=json.dumps({"ok": ok, "tool_name": name, **payload}, ensure_ascii=False),
                      tool_call_id=runtime.tool_call_id)
    update = {"messages": [msg]}
    if items is not None:
        update["items"] = items
    if dropped is not None:
        update["rejected"] = [*(runtime.state.get("rejected") or []), (dropped.name, dropped.region)]
    return Command(update=update)


def make_edit_tools(load_db):
    """수정 tool 3종을 만든다. load_db는 벡터 스토어 로더(테스트에서 주입)."""
    def find(runtime: ToolRuntime, category: str, query: str, meal: str | None = None, near: set[str] | None = None):
        """현재 일정에 없고 사용자가 뺀 적도 없는 후보 중 가장 알맞은 곳 (없으면 None).

        meal='아침'이면 아침 식사 검색어를 더하고, near(이웃 장소의 시군구)가 있으면 같은 시군구 후보를 먼저.
        """
        items: list[ItineraryItem] = runtime.state["items"]
        used = {(i.place.name, i.place.region) for i in items if i.place}
        used |= {tuple(k) for k in runtime.state.get("rejected") or []}
        words = (BREAKFAST_KEYWORDS if meal == "아침" else []) + query.split()
        cands = search_candidates(load_db(), runtime.state["region"], [category], words, k=50)[category]
        cands = [p for p in cands if (p.name, p.region) not in used]
        cands.sort(key=lambda p: bool(near) and p.region not in near)       # 안정 정렬: 같은 시군구 → 검색 순위
        return cands[0] if cands else None

    def neighbors(items: list[ItineraryItem], idx: int) -> set[str]:
        """idx 앞뒤(같은 날) 장소의 시군구 — 교체·추가 후보를 동선 안에서 찾기 위함."""
        day = items[idx].slot.day if idx < len(items) else items[idx - 1].slot.day
        return {items[j].place.region for j in (idx - 1, idx + 1)
                if 0 <= j < len(items) and items[j].place and items[j].slot.day == day}

    @tool("replace_slot", args_schema=ReplaceSlotInput, description=(
        "일정의 특정 슬롯에 배정된 장소를 같은 종류(식당/카페/관광지)의 다른 장소로 교체한다. "
        "사용자가 '바꿔줘', '다른 곳으로', '더 조용한 데로' 처럼 기존 장소 변경을 요청할 때 사용. "
        "slot_id는 [현재 일정]에 표시된 값(예: d1-s2)을 그대로 쓰고, 조건이 있으면 query에 짧은 키워드로 넣는다. "
        "결과 JSON: 성공 {ok:true, data:{slot_id, place}}, 실패 {ok:false, error}."))
    def replace_slot(slot_id: str, query: str, runtime: ToolRuntime) -> Command:
        """슬롯의 장소를 조건에 맞는 미사용 후보로 교체."""
        items: list[ItineraryItem] = runtime.state["items"]
        idx = next((n for n, i in enumerate(items) if i.slot.slot_id == slot_id), None)
        if idx is None:
            return _result(runtime, "replace_slot", False, error=f"없는 slot_id: {slot_id}")
        place = find(runtime, items[idx].slot.category, query, meal=items[idx].slot.meal, near=neighbors(items, idx))
        if place is None:
            return _result(runtime, "replace_slot", False, error="조건에 맞는 다른 장소가 없습니다")
        return _result(runtime, "replace_slot", True, swap_place(items, slot_id, place), dropped=items[idx].place,
                       data={"slot_id": slot_id, "place": place.name})

    @tool("remove_slot", args_schema=RemoveSlotInput, description=(
        "일정에서 특정 슬롯(장소 하나)을 삭제한다. 사용자가 '빼줘', '삭제해줘', '안 갈래' 처럼 제거를 요청할 때 사용. "
        "다른 장소로 바꾸려는 요청이면 이 도구가 아니라 replace_slot을 쓴다. "
        "결과 JSON: 성공 {ok:true, data:{slot_id}}, 실패 {ok:false, error}."))
    def remove_slot(slot_id: str, runtime: ToolRuntime) -> Command:
        """슬롯을 일정에서 제거."""
        items: list[ItineraryItem] = runtime.state["items"]
        if not any(i.slot.slot_id == slot_id for i in items):
            return _result(runtime, "remove_slot", False, error=f"없는 slot_id: {slot_id}")
        removed = next(i.place for i in items if i.slot.slot_id == slot_id)
        return _result(runtime, "remove_slot", True, [i for i in items if i.slot.slot_id != slot_id],
                       dropped=removed, data={"slot_id": slot_id})

    @tool("add_slot", args_schema=AddSlotInput, description=(
        "지정한 일차에 새 장소를 하나 추가한다(관광지·카페는 저녁 식사 앞, 식당은 끝). 사용자가 '하나 더', '추가해줘' 처럼 장소 추가를 요청할 때 사용. "
        "category는 식당·카페·관광지 중 하나, 조건이 있으면 query에 짧은 키워드. "
        "결과 JSON: 성공 {ok:true, data:{slot_id, place}}, 실패 {ok:false, error}."))
    def add_slot(day: int, category: str, query: str, runtime: ToolRuntime) -> Command:
        """그날 일정에 조건에 맞는 미사용 후보를 추가 (관광지·카페는 저녁 앞, 식당은 끝)."""
        items: list[ItineraryItem] = runtime.state["items"]
        same_day = [i for i in items if i.slot.day == day]
        if not same_day:
            return _result(runtime, "add_slot", False, error=f"일정에 {day}일차가 없습니다")
        # 관광지·카페는 그날 저녁 식사 앞에 (저녁 뒤 밤늦게 붙지 않게), 식당은 그날 끝에
        dinner = next((i for i in same_day if i.slot.meal == "저녁"), None)
        pos = items.index(dinner) if dinner and category != "식당" else items.index(same_day[-1]) + 1
        place = find(runtime, category, query, near={items[pos - 1].place.region} if pos and items[pos - 1].place else None)
        if place is None:
            return _result(runtime, "add_slot", False, error="조건에 맞는 장소가 없습니다")
        order = max(i.slot.order for i in same_day) + 1
        new_item = ItineraryItem(slot=Slot(slot_id=f"d{day}-s{order}", day=day, order=order, category=category),
                                 place=place)
        return _result(runtime, "add_slot", True, [*items[:pos], new_item, *items[pos:]],
                       data={"slot_id": new_item.slot.slot_id, "place": place.name})

    return [replace_slot, remove_slot, add_slot]
