"""legacy Directions / Distance Matrix 시그니처를 Routes API로 구현한 googlemaps.Client.

신규 GCP 프로젝트는 legacy API를 켤 수 없어서 기존 코드를 그대로 돌리기 위한 어댑터.
반환 형태는 기존 코드(src/tools.py, src/scheduler)가 읽는 필드만 legacy와 동일하게 맞춘다.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import googlemaps
import requests
from src.logging_config import get_logger

log = get_logger(__name__)

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
MODES = {"transit": "TRANSIT", "walking": "WALK", "driving": "DRIVE", "bicycling": "BICYCLE"}
ROUTE_MASK = ("routes.legs.duration,routes.legs.distanceMeters,routes.legs.startLocation,"
              "routes.legs.endLocation,routes.legs.steps.travelMode,routes.legs.steps.transitDetails")


def _secs(d: str | None) -> int | None:
    """Routes API 기간 문자열('1683s') → 초."""
    return int(d.rstrip("s")) if d else None


def _min_text(s: int) -> str:
    """초 → '28분' / '1시간 5분'."""
    return f"{s // 3600}시간 {(s % 3600) // 60}분" if s >= 3600 else f"{round(s / 60)}분"


def _latlng(loc: dict) -> dict:
    """Routes API 위치 → legacy {'lat','lng'}."""
    ll = (loc or {}).get("latLng", {})
    return {"lat": ll.get("latitude"), "lng": ll.get("longitude")}


class _NetworkError(Exception):
    """일시적 네트워크 오류. 결과 없음(fallback)을 반환하되 캐시하지 않기 위한 내부 신호."""
    def __init__(self, fallback):
        """fallback: 호출부에 돌려줄 '결과 없음' 값."""
        self.fallback = fallback


class RoutesCompatClient(googlemaps.Client):
    """legacy googlemaps.Client 인터페이스를 Routes API로 구현 (디스크 캐시·호출 계측 포함)."""
    calls = 0
    network_calls = 0
    network_seconds = 0.0

    def __init__(self, key: str, **kw):
        """key: Google Maps API 키."""
        super().__init__(key=key, **kw)
        self._api_key = key

    # --- cache ---------------------------------------------------------
    def _cached(self, name: str, args: dict, fetch):
        """요청을 디스크 캐시로 감싼다. 호출 수·네트워크 시간 계측, 네트워크 오류는 캐시하지 않음."""
        RoutesCompatClient.calls += 1
        use_cache = os.getenv("MAPS_CACHE", "1") != "0"
        path = None
        if use_cache:
            h = hashlib.sha1(json.dumps([name, args], ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
            path = Path(os.getenv("MAPS_CACHE_DIR", ".cache/maps")) / f"{name}-{h}.json"
            if path.exists():
                return json.loads(path.read_text())
        RoutesCompatClient.network_calls += 1
        t0 = time.perf_counter()
        try:
            result = fetch()
        except _NetworkError as e:  # 일시 오류(429·타임아웃 등)는 캐시하지 않고 '결과 없음'만 반환
            return e.fallback
        finally:
            RoutesCompatClient.network_seconds += time.perf_counter() - t0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False))
        return result

    def _post(self, url: str, body: dict, mask: str, fallback):
        """Routes API POST. 실패 시 _NetworkError(fallback)."""
        headers = {"X-Goog-Api-Key": self._api_key, "X-Goog-FieldMask": mask,
                   "Content-Type": "application/json"}
        try:
            r = requests.post(url, json=body, headers=headers, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # legacy 코드는 예외 대신 '결과 없음'을 처리하도록 짜여 있음
            log.warning(f"Routes API 실패: {e}")
            raise _NetworkError(fallback)

    # --- geocoding (Geocoding API는 그대로, 캐시만) ----------------------
    def geocode(self, address, **kw):
        """주소 → 좌표 (Geocoding API, 캐시)."""
        return self._cached("geocode", {"address": address, **kw},
                            lambda: super(RoutesCompatClient, self).geocode(address, **kw))

    def reverse_geocode(self, latlng, **kw):
        """좌표 → 주소 (Geocoding API, 캐시)."""
        return self._cached("reverse", {"latlng": list(latlng), **kw},
                            lambda: super(RoutesCompatClient, self).reverse_geocode(latlng, **kw))

    # --- legacy 시그니처 ------------------------------------------------
    def directions(self, origin, destination, mode="driving", departure_time=None, **_):
        # departure_time은 무시: 재현성(캐시 키 고정)과 Routes API 과거 시각 제약 때문
        """legacy directions 형식의 대중교통 경로 (경로 없으면 [])."""
        body = {"origin": {"address": origin}, "destination": {"address": destination},
                "travelMode": MODES.get(mode, "TRANSIT"), "languageCode": "ko"}

        def fetch():
            """실제 API 호출 후 legacy 형식으로 변환."""
            data = self._post(ROUTES_URL, body, ROUTE_MASK, fallback=[]) or {}
            if not data.get("routes"):
                return []
            leg = data["routes"][0]["legs"][0]
            secs = _secs(leg.get("duration")) or 0
            meters = leg.get("distanceMeters", 0)
            steps = []
            for st in leg.get("steps", []):
                if st.get("travelMode") == "TRANSIT":
                    line = st.get("transitDetails", {}).get("transitLine", {})
                    steps.append({"travel_mode": "TRANSIT", "transit_details": {"line": {
                        "short_name": line.get("nameShort"), "name": line.get("name"),
                        "vehicle": {"name": line.get("vehicle", {}).get("name", {}).get("text", "대중교통")}}}})
                else:
                    steps.append({"travel_mode": "WALKING"})
            return [{"legs": [{
                "duration": {"value": secs, "text": _min_text(secs)},
                "distance": {"value": meters, "text": f"{meters / 1000:.1f}km"},
                "start_location": _latlng(leg.get("startLocation")),
                "end_location": _latlng(leg.get("endLocation")),
                "steps": steps}]}]

        return self._cached("directions", body, fetch)

    def distance_matrix(self, origins, destinations, mode="driving", **_):
        """legacy distance_matrix 형식의 소요 시간 행렬 (경로 없으면 ZERO_RESULTS)."""
        body = {"origins": [{"waypoint": {"address": o}} for o in origins],
                "destinations": [{"waypoint": {"address": d}} for d in destinations],
                "travelMode": MODES.get(mode, "TRANSIT")}

        def fetch():
            """실제 API 호출 후 legacy 형식으로 변환."""
            empty = [{"elements": [{"status": "ZERO_RESULTS"} for _ in destinations]} for _ in origins]
            data = self._post(MATRIX_URL, body, "originIndex,destinationIndex,duration,condition",
                              fallback={"rows": empty}) or []
            rows = [{"elements": [{"status": "ZERO_RESULTS"} for _ in destinations]} for _ in origins]
            for e in data:
                secs = _secs(e.get("duration"))
                if e.get("condition") == "ROUTE_EXISTS" and secs is not None:
                    rows[e["originIndex"]]["elements"][e["destinationIndex"]] = {
                        "status": "OK", "duration": {"value": secs, "text": _min_text(secs)}}
            return {"rows": rows}

        return self._cached("matrix", body, fetch)
