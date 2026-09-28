from datetime import date

from eval.new_adapter import run_new_trip, trip_to_request
from src.domain.models import ItineraryItem, Place, Slot

TRIP = {"id": "t", "destination": "부산", "start_location": "부산역", "days": 2,
        "group_type": "친구", "style": "맛집 탐방", "detail": "노포"}


def test_trip_to_request():
    r = trip_to_request(TRIP, date(2026, 10, 1))
    assert r == {"destination": "부산", "start_location": "부산역", "start_date": "2026-10-01", "days": 2,
                 "group_type": "친구", "style": "맛집 탐방", "detail": "노포"}


class App:
    def __init__(self, exc=None):
        self.exc, self.config = exc, None

    async def ainvoke(self, state, config=None):
        self.config = config
        if self.exc:
            raise self.exc
        item = ItineraryItem(slot=Slot(slot_id="d1-s1", day=1, order=1, category="식당"),
                             place=Place(name="A", region="부산광역시 중구", category="식당"), start="12:00")
        return {"items": [item], "final_text": "계획표"}


async def test_run_new_trip_shapes_like_legacy():
    app = App()
    r = await run_new_trip(app, TRIP, date(2026, 10, 1), callbacks=["cb"])
    assert r["ok"] and r["final_message"] == "계획표"
    assert r["itinerary"][0]["name"] == "A" and r["itinerary"][0]["address"] == "부산광역시 중구"
    assert app.config["callbacks"] == ["cb"]


async def test_run_new_trip_records_errors():
    r = await run_new_trip(App(exc=RuntimeError("x")), TRIP, date(2026, 10, 1))
    assert r["ok"] is False and r["error"] == "RuntimeError" and r["itinerary"] == []
