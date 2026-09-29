"""일정 도메인 로직 (외부 의존 없음)."""
from datetime import datetime, timedelta

from src.domain.geo import calculate_haversine_distance
from src.domain.models import ItineraryItem, Place, Slot


# 슬롯 템플릿 (각 날 끝은 숙소 복귀). 당일치기는 첫날, 1박 2일은 첫날+마지막 날, 2박 이상은 중간 날 반복.
#   첫날: 도착 후 점심 → 관광 → 카페 → 저녁 → (밤 관광)
#   중간 날: 아침 → 관광 → 점심 → 관광 → 카페 → 저녁 → (밤 관광)
#   (밤 관광)은 밤에 가야 좋은 곳(timefit)이 근처에 있을 때만 남는다
#   마지막 날: 아침 → 관광 → 점심 → 귀가 (귀가 시각을 넘기면 잘림)
_FIRST = [("식당", "점심"), ("관광지", None), ("카페", None), ("식당", "저녁"), ("관광지", "밤")]
_MIDDLE = [("식당", "아침"), ("관광지", None), ("식당", "점심"), ("관광지", None), ("카페", None), ("식당", "저녁"),
           ("관광지", "밤")]
_LAST = [("식당", "아침"), ("관광지", None), ("식당", "점심")]
DAY_START = "09:00"          # 2일차부터 하루 시작(숙소 출발)
BREAKFAST_KEYWORDS = ["아침", "해장국", "브런치"]   # 아침 식사 슬롯 검색어 (초안·수정 공통)


def build_slots(days: int) -> list[Slot]:
    """여행 일수에 맞는 슬롯 목록 (템플릿 순서 고정)."""
    slots = []
    for d in range(1, days + 1):
        template = _FIRST if d == 1 else _LAST if d == days else _MIDDLE
        for i, (cat, tag) in enumerate(template, start=1):
            when = tag if tag == "밤" else None
            slots.append(Slot(slot_id=f"d{d}-s{i}", day=d, order=i, category=cat, meal=None if when else tag, when=when))
    return slots


MAX_KM = {"transit": 5.0, "driving": 15.0}   # 직전 장소에서 이 거리(직선) 안의 후보만 — 없으면 해제
RANK_KM = 0.3                                # 검색 순위 한 칸 = 0.3km: 순위를 조금 양보하고 가까운 곳을 고르는 비율
N_ALTERNATIVES = 3


def candidate_key(slot: Slot) -> str:
    """슬롯이 쓸 후보 목록 키: 아침 식사는 아침용 검색 결과('아침'), 나머지는 카테고리."""
    return "아침" if slot.meal == "아침" else slot.category


def _fits_time(slot: Slot, p: Place) -> bool:
    """밤 슬롯엔 밤에 가야 좋은 곳만, 낮 관광 슬롯엔 그런 곳을 넣지 않는다 (밤 슬롯 몫으로 남김)."""
    night = p.time_fit.get("밤", 0.0) >= NIGHT_FIT
    if slot.when == "밤":
        return night
    return not (night and slot.category == "관광지")


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
        key = candidate_key(s) if candidate_key(s) in candidates else s.category
        pool = [(i, p) for i, p in enumerate(candidates.get(key, [])) if (p.name, p.region) not in used
                and _fits_time(s, p)]
        if anchors is None:
            ranked = [p for _, p in pool]
        else:
            scored = sorted(((i * RANK_KM + _km(prev, p, max_km), _km(prev, p, max_km) <= max_km, p) for i, p in pool),
                            key=lambda x: (not x[1], x[0]))
            ranked = [p for *_, p in scored]
        if s.when == "밤" and anchors is not None:     # 밤에는 멀리 가지 않는다 — 반경 안에 없으면 비움
            ranked = [p for p in ranked if _km(prev, p, max_km) <= max_km]
        pick = ranked[0] if ranked else None
        picks[s.slot_id], alts[s.slot_id] = pick, ranked[1:1 + N_ALTERNATIVES]
        if pick:
            used.add((pick.name, pick.region))
            if pick.lat is not None:
                prev = (pick.lat, pick.lng)
    if anchors is not None:
        _refine(slots, candidates, anchors, picks, max_km)
    return picks, alts


def _xy(p: Place | None) -> tuple[float, float] | None:
    return (p.lat, p.lng) if p is not None and p.lat is not None else None


def _refine(slots: list[Slot], candidates: dict[str, list[Place]], anchors: dict[int, tuple[float, float]],
            picks: dict[str, Place | None], max_km: float) -> None:
    """지그재그 보정 (한 바퀴): 각 슬롯을 `순위×RANK_KM + 앞 이웃 거리 + 뒤 이웃 거리`가 최소인 후보로 다시 고른다.

    탐욕 배정은 직전 장소만 봐서, 하루 첫 장소가 앵커 반대편(북쪽)에 뽑히고 나머지가 남쪽에 몰리는 식의 되돌아가기가 생긴다.
    """
    for n, s in enumerate(slots):
        if picks.get(s.slot_id) is None:
            continue
        same = [x for x in slots if x.day == s.day]
        k = same.index(s)
        before = next((_xy(picks.get(x.slot_id)) for x in reversed(same[:k]) if _xy(picks.get(x.slot_id))), None)
        prev = before or anchors.get(s.day)
        nxt = next((_xy(picks.get(x.slot_id)) for x in same[k + 1:] if _xy(picks.get(x.slot_id))), None)
        taken = {(p.name, p.region) for sid, p in picks.items() if p is not None and sid != s.slot_id}
        key = candidate_key(s) if candidate_key(s) in candidates else s.category
        pool = [(i, p) for i, p in enumerate(candidates.get(key, [])) if (p.name, p.region) not in taken
                and _fits_time(s, p)]

        def cost(i: int, p: Place) -> tuple[bool, float]:
            d_prev = _km(prev, p, max_km)
            return d_prev > max_km, i * RANK_KM + d_prev + (_km(nxt, p, max_km) if nxt else 0.0)
        best = min(pool, key=lambda ip: cost(*ip), default=None)
        if best is not None:
            picks[s.slot_id] = best[1]


def swap_place(items: list[ItineraryItem], slot_id: str, place: Place) -> list[ItineraryItem]:
    """slot_id의 장소를 place로 교체 (사유 초기화, 대안 목록에서 제거). 시간표는 호출부가 다시 계산."""
    return [i.model_copy(update={"place": place, "reason": "",
                                 "alternatives": [a for a in i.alternatives if (a.name, a.region) != (place.name, place.region)]})
            if i.slot.slot_id == slot_id else i for i in items]


STAY_MIN = {"식당": 90, "카페": 60, "관광지": 120}
BREAKFAST_MIN = 60


MEAL_WINDOW = {"아침": ("08:00", "10:00"), "점심": ("11:30", "14:00"), "저녁": ("17:30", "20:00")}  # 식사 시작 시각
WHEN_WINDOW = {"밤": ("19:00", "21:00")}     # 밤 관광 시작 시각
NIGHT_FIT = 0.10             # 밤 쪽 리뷰 비율이 이 이상이면 '밤에 가야 좋은 곳' (retrieval/timefit.py 검증값)
NIGHT_MIN = 60
DAY_END = "22:00"            # 이 시각을 넘겨 끝나는 장소는 넣지 않음 (초안 생성 시)


def _at(hhmm: str) -> datetime:
    """'HH:MM' → 기준일의 datetime (자정을 넘긴 시각과 비교할 수 있게)."""
    return datetime(2000, 1, 1, *map(int, hhmm.split(":")))


MIN_SAME_DAY_MIN = 90       # 당일치기: 도착~출발 사이에 장소 한 곳(식당 90분)은 들어가야 함


def check_trip_times(days: int, start_time: str, end_time: str) -> str | None:
    """입력 시각 검증. 일정이 통째로 비게 되는 조합이면 사용자에게 보여줄 안내, 아니면 None."""
    if days == 1 and _at(end_time) < _at(start_time) + timedelta(minutes=MIN_SAME_DAY_MIN):
        return f"당일치기는 출발 시각이 도착 시각보다 {MIN_SAME_DAY_MIN}분 이상 늦어야 일정을 넣을 수 있어요."
    last_day_first_end = _at(DAY_START) + timedelta(minutes=BREAKFAST_MIN)
    if days > 1 and _at(end_time) < last_day_first_end:
        return f"마지막 날 출발이 {last_day_first_end:%H:%M} 이전이면 그날 일정을 넣을 수 없어요."
    return None


def _next_meal(items: list[ItineraryItem], idx: int) -> ItineraryItem | None:
    """idx 다음, 같은 날의 첫 식사 슬롯."""
    for it in items[idx + 1:]:
        if it.slot.day != items[idx].slot.day:
            return None
        if it.place is not None and it.slot.meal in MEAL_WINDOW:
            return it
    return None


def assign_times(items: list[ItineraryItem], legs: dict[str, tuple[int, str | None]], default_move: int = 30,
                 first_start: str = "12:00", last_end: str | None = None, last_day: int | None = None,
                 prune: bool = False, first_legs: dict[str, tuple[int, str | None]] | None = None
                 ) -> list[ItineraryItem]:
    """일차별 시작(1일차 first_start, 이후 DAY_START)부터 체류·이동 시간을 누적. legs: slot_id → (이동 분, 교통편).

    식사는 MEAL_WINDOW 시작 전이면 그때까지 기다린다(10시 도착해도 점심은 11:30).
    prune(초안 생성): 시간대를 놓친 식사, 넣으면 다음 식사가 시간대를 넘기는 관광·카페, DAY_END 이후 장소를 뺀다.
    last_day의 장소 중 last_end(귀가 출발 시각)를 넘겨 끝나는 곳부터 그날 나머지를 뺀다.
    사용자 수정 뒤(prune=False)에는 장소를 지우지 않는다.
    first_legs: 하루 첫 장소의 숙소(첫날은 도착지)→장소 이동. 하루 시작 시각은 숙소 출발 시각이다.
    """
    first_legs = first_legs or {}
    out, cursor, day, prev_placed, closed = [], None, None, False, False
    for idx, it in enumerate(items):
        if it.slot.day != day:
            day, prev_placed, closed = it.slot.day, False, False
            cursor = _at(first_start) if day == 1 else _at(DAY_START)
        if closed:
            continue
        if it.place is None:
            out.append(it)
            continue
        update, at = {}, cursor
        if prev_placed:
            mv, detail = legs.get(it.slot.slot_id, (default_move, None))
            at = cursor + timedelta(minutes=mv)
            update.update(move_min=mv, move_detail=detail)
        elif it.slot.slot_id in first_legs:
            mv, detail = first_legs[it.slot.slot_id]
            at = cursor + timedelta(minutes=mv)
            update.update(move_min=mv, move_detail=detail, from_lodging=True)
        window = MEAL_WINDOW.get(it.slot.meal) or WHEN_WINDOW.get(it.slot.when)
        if window:
            at = max(at, _at(window[0]))
        stay = BREAKFAST_MIN if it.slot.meal == "아침" else NIGHT_MIN if it.slot.when == "밤" \
            else STAY_MIN.get(it.slot.category, 90)
        end = at + timedelta(minutes=stay)
        if prune:
            if window and at > _at(window[1]):
                continue                              # 식사 시간대를 놓침 (늦은 도착)
            nxt = _next_meal(items, idx)
            if not window and nxt and end + timedelta(minutes=default_move) > _at(MEAL_WINDOW[nxt.slot.meal][1]):
                continue                              # 이걸 넣으면 다음 식사가 시간대를 넘김
            if end > _at(DAY_END):
                continue
        if last_end and day == last_day and end > _at(last_end):
            closed = True                             # 귀가 시각을 넘김 → 이 장소부터 그날 나머지 제외
            continue
        update.update(start=at.strftime("%H:%M"), end=end.strftime("%H:%M"))
        out.append(it.model_copy(update=update))
        cursor, prev_placed = end, True
    return out
