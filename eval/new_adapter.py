"""새 워크플로우(src.agent.workflow)를 평가 러너에서 호출. 반환 형식은 run_legacy_trip과 동일."""
from datetime import date


def trip_to_request(trip: dict, start: date) -> dict:
    return {"destination": trip["destination"], "start_location": trip.get("start_location", ""),
            "start_date": start.isoformat(), "days": trip["days"], "group_type": trip["group_type"],
            "style": trip["style"], "detail": trip.get("detail", "")}


async def run_new_trip(app, trip: dict, start: date, callbacks: list | None = None) -> dict:
    from src.agent.workflow import to_legacy_itinerary
    try:
        out = await app.ainvoke({"request": trip_to_request(trip, start)},
                                config={"callbacks": callbacks or [], "recursion_limit": 50})
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "detail": str(e)[:300], "itinerary": [], "final_message": ""}
    return {"ok": True, "error": None, "itinerary": to_legacy_itinerary(out["items"]),
            "final_message": out.get("final_text", "")}
