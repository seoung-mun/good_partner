import pytest
from src.services import maps_client as maps_compat
from src.services.maps_client import RoutesCompatClient

KEY = "AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAK"

ROUTE = {"routes": [{"legs": [{
    "duration": "1683s", "distanceMeters": 4239,
    "startLocation": {"latLng": {"latitude": 37.57, "longitude": 126.97}},
    "endLocation": {"latLng": {"latitude": 37.55, "longitude": 126.98}},
    "steps": [
        {"travelMode": "WALK"},
        {"travelMode": "TRANSIT", "transitDetails": {"transitLine": {
            "nameShort": "402", "name": "간선402", "vehicle": {"name": {"text": "버스"}}}}},
    ]}]}]}


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPS_CACHE_DIR", str(tmp_path))
    RoutesCompatClient.calls = RoutesCompatClient.network_calls = 0
    return RoutesCompatClient(key=KEY)


def test_directions_legacy_shape(client, monkeypatch):
    monkeypatch.setattr(maps_compat.requests, "post", lambda *a, **k: FakeResp(ROUTE))
    leg = client.directions("경복궁", "N서울타워", mode="transit")[0]["legs"][0]
    assert leg["duration"] == {"value": 1683, "text": "28분"}
    assert leg["distance"]["text"] == "4.2km"
    assert leg["start_location"] == {"lat": 37.57, "lng": 126.97}
    assert leg["steps"][0]["travel_mode"] == "WALKING"
    line = leg["steps"][1]["transit_details"]["line"]
    assert line["short_name"] == "402" and line["vehicle"]["name"] == "버스"


def test_directions_no_route_returns_empty_list(client, monkeypatch):
    monkeypatch.setattr(maps_compat.requests, "post", lambda *a, **k: FakeResp({}))
    assert client.directions("A", "B", mode="walking") == []


def test_distance_matrix_legacy_shape(client, monkeypatch):
    rows = [
        {"originIndex": 0, "destinationIndex": 1, "duration": "600s", "condition": "ROUTE_EXISTS"},
        {"originIndex": 1, "destinationIndex": 0, "condition": "ROUTE_NOT_FOUND"},
        {"originIndex": 0, "destinationIndex": 0, "condition": "ROUTE_NOT_FOUND"},
        {"originIndex": 1, "destinationIndex": 1, "condition": "ROUTE_NOT_FOUND"},
    ]
    monkeypatch.setattr(maps_compat.requests, "post", lambda *a, **k: FakeResp(rows))
    m = client.distance_matrix(["A", "B"], ["A", "B"], mode="transit")
    assert m["rows"][0]["elements"][1]["duration"]["value"] == 600
    assert m["rows"][1]["elements"][0] == {"status": "ZERO_RESULTS"}


def test_cache_avoids_second_network_call(client, monkeypatch):
    hits = []
    monkeypatch.setattr(maps_compat.requests, "post", lambda *a, **k: hits.append(1) or FakeResp(ROUTE))
    client.directions("A", "B", mode="transit")
    client.directions("A", "B", mode="transit")
    assert len(hits) == 1
    assert RoutesCompatClient.calls == 2 and RoutesCompatClient.network_calls == 1


def test_http_error_returns_empty_not_raise(client, monkeypatch):
    monkeypatch.setattr(maps_compat.requests, "post", lambda *a, **k: FakeResp({"error": {}}, 403))
    assert client.directions("A", "B", mode="transit") == []


def test_http_error_is_not_cached(client, monkeypatch):
    hits = []
    monkeypatch.setattr(maps_compat.requests, "post",
                        lambda *a, **k: hits.append(1) or FakeResp({"error": {}}, 429))
    client.directions("A", "B", mode="transit")
    client.directions("A", "B", mode="transit")
    assert len(hits) == 2          # 실패 응답은 캐시하지 않고 다음 호출에서 재시도


def test_real_no_route_is_cached(client, monkeypatch):
    hits = []
    monkeypatch.setattr(maps_compat.requests, "post", lambda *a, **k: hits.append(1) or FakeResp({}))
    client.directions("A", "B", mode="transit")
    client.directions("A", "B", mode="transit")
    assert len(hits) == 1          # 정상 200 '경로 없음'은 캐시


def test_network_time_is_accumulated(client, monkeypatch):
    RoutesCompatClient.network_seconds = 0.0
    monkeypatch.setattr(maps_compat.requests, "post", lambda *a, **k: FakeResp(ROUTE))
    client.directions("C", "D", mode="transit")
    assert RoutesCompatClient.network_seconds > 0
