"""수정 에이전트 tool. 얇은 래퍼: 입력 검증(args_schema) → 도메인·검색 함수 → 상태 갱신(Command) + 결과 envelope."""
import json
from typing import Literal

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.domain.models import ItineraryItem, Slot
from src.retrieval.retriever import search_candidates


class ReplaceSlotInput(BaseModel):
    slot_id: str = Field(description="바꿀 일정 슬롯 ID (예: d1-s2)")
    query: str = Field(description="새 장소 조건 키워드 (예: 조용한, 바다뷰). 조건이 없으면 빈 문자열")


class RemoveSlotInput(BaseModel):
    slot_id: str = Field(description="삭제할 일정 슬롯 ID (예: d1-s2)")


class AddSlotInput(BaseModel):
    day: int = Field(description="추가할 일차 (1부터)")
    category: Literal["식당", "카페", "관광지"] = Field(description="추가할 장소 종류")
    query: str = Field(description="장소 조건 키워드. 없으면 빈 문자열")


def _result(runtime: ToolRuntime, name: str, ok: bool, items=None, **payload) -> Command:
    msg = ToolMessage(content=json.dumps({"ok": ok, "tool_name": name, **payload}, ensure_ascii=False),
                      tool_call_id=runtime.tool_call_id)
    update = {"messages": [msg]}
    if items is not None:
        update["items"] = items
    return Command(update=update)


def make_edit_tools(load_db):
    def find(runtime: ToolRuntime, category: str, query: str):
        items: list[ItineraryItem] = runtime.state["items"]
        used = {(i.place.name, i.place.region) for i in items if i.place}
        cands = search_candidates(load_db(), runtime.state["region"], [category], query.split(), k=50)[category]
        return next((p for p in cands if (p.name, p.region) not in used), None)

    @tool(args_schema=ReplaceSlotInput)
    def replace_slot(slot_id: str, query: str, runtime: ToolRuntime) -> Command:
        """일정의 한 슬롯을 같은 종류의 다른 장소로 교체합니다."""
        items: list[ItineraryItem] = runtime.state["items"]
        idx = next((n for n, i in enumerate(items) if i.slot.slot_id == slot_id), None)
        if idx is None:
            return _result(runtime, "replace_slot", False, error=f"없는 slot_id: {slot_id}")
        place = find(runtime, items[idx].slot.category, query)
        if place is None:
            return _result(runtime, "replace_slot", False, error="조건에 맞는 다른 장소가 없습니다")
        new = [*items]
        new[idx] = items[idx].model_copy(update={"place": place, "reason": ""})
        return _result(runtime, "replace_slot", True, new, data={"slot_id": slot_id, "place": place.name})

    @tool(args_schema=RemoveSlotInput)
    def remove_slot(slot_id: str, runtime: ToolRuntime) -> Command:
        """일정에서 슬롯 하나를 삭제합니다."""
        items: list[ItineraryItem] = runtime.state["items"]
        if not any(i.slot.slot_id == slot_id for i in items):
            return _result(runtime, "remove_slot", False, error=f"없는 slot_id: {slot_id}")
        return _result(runtime, "remove_slot", True, [i for i in items if i.slot.slot_id != slot_id],
                       data={"slot_id": slot_id})

    @tool(args_schema=AddSlotInput)
    def add_slot(day: int, category: str, query: str, runtime: ToolRuntime) -> Command:
        """지정한 일차의 마지막에 장소를 하나 추가합니다."""
        items: list[ItineraryItem] = runtime.state["items"]
        same_day = [i for i in items if i.slot.day == day]
        if not same_day:
            return _result(runtime, "add_slot", False, error=f"일정에 {day}일차가 없습니다")
        place = find(runtime, category, query)
        if place is None:
            return _result(runtime, "add_slot", False, error="조건에 맞는 장소가 없습니다")
        order = max(i.slot.order for i in same_day) + 1
        new_item = ItineraryItem(slot=Slot(slot_id=f"d{day}-s{order}", day=day, order=order, category=category),
                                 place=place)
        pos = items.index(same_day[-1]) + 1
        return _result(runtime, "add_slot", True, [*items[:pos], new_item, *items[pos:]],
                       data={"slot_id": new_item.slot.slot_id, "place": place.name})

    return [replace_slot, remove_slot, add_slot]
