from src.services import weather


class Resp:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class Geo:
    def geocode(self, q, language="ko"):
        return [{"geometry": {"location": {"lat": 37.75, "lng": 128.9}}}] if "강릉" in q else []


def test_falls_back_to_google_geocoding_when_owm_cannot_find(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None):
        if "geo/1.0" in url:
            return Resp([])                                   # OWM은 '강원특별자치도 강릉시'를 모름
        seen.update(params)
        return Resp({"list": [{"dt_txt": "2026-10-01 03:00:00", "main": {"temp": 21.0}, "weather": [{"description": "맑음"}]}]})
    monkeypatch.setenv("OWM_API_KEY", "k")
    monkeypatch.setattr(weather.requests, "get", fake_get)
    monkeypatch.setattr(weather, "_google_client", lambda: Geo())
    out = weather.forecast_summary("강원특별자치도 강릉시", "2026년 10월 01일")
    assert (seen["lat"], seen["lon"]) == (37.75, 128.9)
    assert "맑음" in out and not out.startswith("오류")
