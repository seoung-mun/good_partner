"""일정 도메인 로직 (외부 의존 없음)."""
from datetime import datetime, timedelta

from src.domain.geo import calculate_haversine_distance
from src.domain.models import ItineraryItem, Place, Slot


# 슬롯 템플릿: 1일차 도착 후 점심부터 4곳, 중간 날 10시 시작 5곳, 마지막 날 관광·점심·카페 (귀가 시각 넘으면 잘림)
_FIRST = [("식당", "점심"), ("카페", None), ("관광지", None), ("식당", "저녁")]
_MIDDLE = [("관광지", None), ("식당", "점심"), ("카페", None), ("관광지", None), ("식당", "저녁")]
_LAST = [("관광지", None), ("식당", "점심"), ("카페", None)]


def build_slots(days: int) -> list[Slot]:
    """여행 일수에 맞는 슬롯 목록 (템플릿 순서 고정)."""
    slots = []
    for d in range(1, days + 1):
        template = _FIRST if d == 1 else _LAST if d == days else _MIDDLE
        for i, (cat, meal) in enumerate(template, start=1):
            slots.append(Slot(slot_id=f"d{d}-s{i}", day=d, order=i, category=cat, meal=meal))
    return slots


MAX_KM = {"transit": 5.0, "driving": 15.0}   # 직전 장소에서 이 거리(직선) 안의 후보만 — 없으면 해제
RANK_KM = 0.3                                # 검색 순위 한 칸 = 0.3km: 순위를 조금 양보하고 가까운 곳을 고르는 비율
N_ALTERNATIVES = 3


def _km(a: tuple[float, float] | None, p: Place, unknown: float) -> float:
    """직전 위치 → 후보 직선 거리. 어느 쪽이든 좌표가 없으면 unknown(반경 경계로 취급)."""
    if a is None or p.lat is None or p.lng is None:
        return unknown
    return calculate_haversine_distance(a[0], a[1], p.lat, p.lng)


def assign_places(slots: list[Slot], candidates: dict[str, list[Place]],
                  anchors: dict[int, tuple[float, float]] | None = None,
                  max_km: float = MAX_KM["transit"]) -> tuple[dict[str, Place | None], dict[str, list[Place]]]:
    """슬롯 순서대로 장소 배정 → (slot_id → 장소, slot_id → 대안 후보).

    anchors(일차 → 출발 좌표)가 있으면 동선 기준: 직전 장소(하루 시작은 앵커)에서 max_km 안의 후보 중
    `순위 × RANK_KM + 거리`가 가장 작은 곳. 반경 안에 없으면 전체에서 고른다. 없으면 검색 순위순.
    한 일정 안에서 같은 장소는 한 번만.
    """
    used, picks, alts = set(), {}, {}
    prev, day = None, None
    for s in slots:
        if s.day != day:
            day = s.day
            prev = (anchors or {}).get(day)
        pool = [(i, p) for i, p in enumerate(candidates.get(s.category, [])) if (p.name, p.region) not in used]
        if anchors is None:
            ranked = [p for _, p in pool]
        else:
            scored = sorted(((i * RANK_KM + _km(prev, p, max_km), _km(prev, p, max_km) <= max_km, p) for i, p in pool),
                            key=lambda x: (not x[1], x[0]))
            ranked = [p for *_, p in scored]
        pick = ranked[0] if ranked else None
        picks[s.slot_id], alts[s.slot_id] = pick, ranked[1:1 + N_ALTERNATIVES]
        if pick:
            used.add((pick.name, pick.region))
            if pick.lat is not None:
                prev = (pick.lat, pick.lng)
    return picks, alts


def swap_place(items: list[ItineraryItem], slot_id: str, place: Place) -> list[ItineraryItem]:
    """slot_id의 장소를 place로 교체 (사유 초기화, 대안 목록에서 제거). 시간표는 호출부가 다시 계산."""
    return [i.model_copy(update={"place": place, "reason": "",
                                 "alternatives": [a for a in i.alternatives if (a.name, a.region) != (place.name, place.region)]})
            if i.slot.slot_id == slot_id else i for i in items]


STAY_MIN = {"식당": 90, "카페": 60, "관광지": 120}


def assign_times(items: list[ItineraryItem], legs: dict[str, tuple[int, str | None]], default_move: int = 30,
                 first_start: str = "12:00", last_end: str | None = None, last_day: int | None = None
                 ) -> list[ItineraryItem]:
    """일차별 시작(1일차 first_start, 이후 10:00)부터 체류·이동 시간을 누적. legs: slot_id → (이동 분, 교통편).

    last_day의 장소 중 last_end(귀가 출발 시각)를 넘겨 끝나는 곳은 일정에서 뺀다.
    """
    h, m = map(int, first_start.split(":"))
    out, cursor, day, prev_placed, closed = [], None, None, False, False
    for it in items:
        if it.slot.day != day:
            day, prev_placed = it.slot.day, False
            cursor = datetime(2000, 1, 1, h, m) if day == 1 else datetime(2000, 1, 1, 10, 0)
        if closed and day == last_day:
            continue
        if it.place is None:
            out.append(it)
            continue
        update, at = {}, cursor
        if prev_placed:
            mv, detail = legs.get(it.slot.slot_id, (default_move, None))
            at = cursor + timedelta(minutes=mv)
            update.update(move_min=mv, move_detail=detail)
        end = at + timedelta(minutes=STAY_MIN.get(it.slot.category, 90))
        if last_end and day == last_day and end.strftime("%H:%M") > last_end:
            closed = True                             # 귀가 시각을 넘김 → 이 장소부터 그날 나머지 제외
            continue
        update.update(start=at.strftime("%H:%M"), end=end.strftime("%H:%M"))
        out.append(it.model_copy(update=update))
        cursor, prev_placed = end, True
    return out
