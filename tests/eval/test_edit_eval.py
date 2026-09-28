import json

from langchain_core.messages import ToolMessage

from eval.run_edit import check_success, from_legacy_itinerary, scenarios, tool_outcomes
from src.domain.models import ItineraryItem, Place, Slot

LEGACY = [{"type": "식당", "name": "A", "address": "부산광역시 중구", "day": 1, "slot_id": "d1-s1", "start": "12:00"},
          {"type": "move", "duration_min": 5, "day": 1},
          {"type": "카페", "name": "B", "address": "부산광역시 중구", "day": 1, "slot_id": "d1-s2", "start": "13:35"},
          {"type": "관광지", "name": "C", "address": "부산광역시 중구", "day": 1, "slot_id": "d1-s3"}]


def test_from_legacy_itinerary():
    items = from_legacy_itinerary(LEGACY)
    assert [(i.slot.slot_id, i.slot.category, i.place.name) for i in items] == [
        ("d1-s1", "식당", "A"), ("d1-s2", "카페", "B"), ("d1-s3", "관광지", "C")]
    assert items[0].slot.meal == "점심"


def test_scenarios_and_success_checks():
    before = from_legacy_itinerary(LEGACY)
    sc = {s["kind"]: s for s in scenarios(before)}
    assert set(sc) == {"replace", "remove", "add", "natural"}
    replaced = [i.model_copy(update={"place": Place(name="Z", region="r", category="카페")}) if i.slot.slot_id == "d1-s2" else i for i in before]
    assert check_success(sc["replace"], before, replaced)
    assert not check_success(sc["replace"], before, before)
    assert check_success(sc["remove"], before, [i for i in before if i.slot.slot_id != sc["remove"]["slot_id"]])
    added = before + [ItineraryItem(slot=Slot(slot_id="d1-s4", day=1, order=4, category="카페"),
                                    place=Place(name="N", region="r", category="카페"))]
    assert check_success(sc["add"], before, added)
    lunch = [i.model_copy(update={"place": Place(name="Q", region="r", category="식당")}) if i.slot.slot_id == "d1-s1" else i for i in before]
    assert check_success(sc["natural"], before, lunch)


def test_tool_outcomes_counts_envelopes():
    msgs = [ToolMessage(content=json.dumps({"ok": True}), tool_call_id="1"),
            ToolMessage(content=json.dumps({"ok": False, "error": "x"}), tool_call_id="2"),
            ToolMessage(content="not json", tool_call_id="3")]
    assert tool_outcomes(msgs) == {"tool_results": 3, "tool_failures": 2}
