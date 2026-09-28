"""수정 에이전트 시나리오 평가: 최종 E2E 일정 위에 수정 요청 4종 → 기대 변경 여부, tool 오류·실패, 무효 호출, 지연."""
import argparse
import asyncio
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from src.domain.models import ItineraryItem, Place, Slot


def _meal(day: int, order: int, category: str):
    if category != "식당":
        return None
    lunch, dinner = (1, 4) if day == 1 else (2, 5)
    return "점심" if order == lunch else "저녁" if order == dinner else None


def from_legacy_itinerary(itinerary: list[dict]) -> list[ItineraryItem]:
    items = []
    for x in itinerary:
        if x.get("type") == "move" or not x.get("slot_id"):
            continue
        day, order = int(x["day"]), int(x["slot_id"].split("-s")[1])
        items.append(ItineraryItem(
            slot=Slot(slot_id=x["slot_id"], day=day, order=order, category=x["type"], meal=x["meal"] if "meal" in x else _meal(day, order, x["type"])),
            place=Place(name=x["name"], region=x.get("address", ""), category=x["type"], reviews=x.get("reviews", [])),
            start=x.get("start"), end=x.get("end"), reason=x.get("description", "")))
    return items


def scenarios(items: list[ItineraryItem]) -> list[dict]:
    placed = [i for i in items if i.place]
    lunch = next(i for i in placed if i.slot.meal == "점심")
    replace_t = next(i for i in placed if i is not lunch)
    remove_t = next(i for i in reversed(placed) if i not in (lunch, replace_t))
    return [
        {"kind": "replace", "slot_id": replace_t.slot.slot_id, "text": f"{replace_t.slot.slot_id} 장소를 다른 곳으로 바꿔줘"},
        {"kind": "remove", "slot_id": remove_t.slot.slot_id, "text": f"{remove_t.slot.slot_id} 일정은 빼줘"},
        {"kind": "add", "day": 1, "category": "카페", "text": "1일차에 카페 하나 더 추가해줘"},
        {"kind": "natural", "slot_id": lunch.slot.slot_id, "text": f"{lunch.slot.day}일차 점심 식당을 다른 곳으로 바꿔줘"},
    ]


def check_success(sc: dict, before: list[ItineraryItem], after: list[ItineraryItem]) -> bool:
    by = lambda items: {i.slot.slot_id: i for i in items}
    b, a = by(before), by(after)
    if sc["kind"] in ("replace", "natural"):
        new = a.get(sc["slot_id"])
        return bool(new and new.place and new.place.name != b[sc["slot_id"]].place.name and len(a) == len(b))
    if sc["kind"] == "remove":
        return sc["slot_id"] not in a and len(a) == len(b) - 1
    count = lambda items: sum(i.slot.day == sc["day"] and i.slot.category == sc["category"] and i.place is not None
                              for i in items)
    return count(after) == count(before) + 1 and len(after) == len(before) + 1


def tool_outcomes(messages) -> dict:
    from langchain_core.messages import ToolMessage
    results = [m for m in messages if isinstance(m, ToolMessage)]
    fails = 0
    for m in results:
        try:
            fails += not json.loads(m.content).get("ok", False)
        except (ValueError, AttributeError):
            fails += 1
    return {"tool_results": len(results), "tool_failures": fails}


async def main(args):
    load_dotenv(".env")
    from eval.instrument import configure_caches
    from eval.metrics import aggregate
    from eval.new_adapter import trip_to_request
    from eval.trace import AgentTracer
    from src.agent.edit_agent import build_edit_app, edit
    from src.domain.region import resolve_destination
    from src.services.maps import resolve_admin_region

    cache_state = configure_caches(args.no_cache)
    app = build_edit_app()
    e2e = json.loads(Path(args.e2e).read_text())
    trips = [r for r in e2e["rows"] if r["ok"] and r["n_places"] >= 3][: args.limit]
    start = date.today() + timedelta(days=1)
    rows = []
    for trip in trips:
        items = from_legacy_itinerary(trip["itinerary"])
        resolved = resolve_destination(trip["destination"])
        region = " ".join(x for x in resolved if x) if resolved else await resolve_admin_region(trip["destination"], trip["destination"])
        state = {"request": trip_to_request(trip, start), "region": region, "items": items}
        for sc in scenarios(items):
            tracer = AgentTracer(known_tools={"replace_slot", "remove_slot", "add_slot"})
            t0 = time.time()
            try:
                out = await edit(app, state, sc["text"], config={"callbacks": [tracer]})
                ok, after, msgs, err = True, out["items"], out["messages"], None
            except Exception as e:
                ok, after, msgs, err = False, items, [], f"{type(e).__name__}: {str(e)[:200]}"
            tr = tracer.summary()
            to = tool_outcomes(msgs)
            row = {"trip_id": trip["id"], "kind": sc["kind"], "text": sc["text"], "run_ok": ok, "error": err,
                   "success": float(ok and check_success(sc, items, after)), "latency_s": time.time() - t0,
                   **{k: tr[k] for k in ("llm_calls", "tool_calls", "tool_errors", "tool_calls_requested",
                                         "invalid_tool_calls", "redundant_tool_calls")},
                   **to, "tool_failure_rate": (to["tool_failures"] / to["tool_results"]) if to["tool_results"] else None,
                   "no_tool_call": float(tr["tool_calls_requested"] == 0),
                   "invalid_tool_call_rate": tr["invalid_tool_call_rate"]}
            rows.append(row)
            print(f"{trip['id']} {sc['kind']:8s} success={row['success']:.0f} tools={tr['tool_calls']} "
                  f"fail={to['tool_failures']} invalid={tr['invalid_tool_calls']} llm={tr['llm_calls']} {row['latency_s']:.0f}s",
                  flush=True)
    keys = ["success", "latency_s", "llm_calls", "tool_calls", "tool_failure_rate", "invalid_tool_call_rate",
            "no_tool_call", "redundant_tool_calls"]
    out = {"llm_model": os.environ["LLM_MODEL"], "cache": cache_state, "summary": aggregate(rows, keys, by="kind"), "rows": rows}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="eval/results/e2e_final.json")
    ap.add_argument("--out", default="eval/results/edit_final.json")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--no-cache", action="store_true")
    asyncio.run(main(ap.parse_args()))
