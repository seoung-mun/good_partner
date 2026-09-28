"""일정 도메인 로직 (외부 의존 없음)."""
from itertools import permutations

from datetime import datetime, timedelta

from src.domain.models import ItineraryItem, Place, Slot


def _solve_tsp(duration_matrix, start_fixed, n):
    """TSP 알고리즘"""
    min_duration = float('inf')
    best_order_indices = []
    
    indices = list(range(n))
    if start_fixed: indices = list(range(1, n))

    if len(indices) > 8:
        current = 0
        unvisited = set(indices)
        path = [0]
        cost = 0
        while unvisited:
            nxt = min(unvisited, key=lambda i: duration_matrix[current][i])
            cost += duration_matrix[current][nxt]
            path.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return path, cost

    for p in permutations(indices):
        current_indices = [0] + list(p) if start_fixed else list(p)
        current_dur = sum(duration_matrix[current_indices[i]][current_indices[i+1]] for i in range(len(current_indices)-1))
        if current_dur < min_duration:
            min_duration = current_dur
            best_order_indices = current_indices
            
    return best_order_indices, min_duration


# 기존 graph_flow 규칙: 1일차 12시 점심 시작 4곳, 중간 날 10시 시작 5곳, 마지막 날 관광 1곳
_FIRST = [("식당", "점심"), ("카페", None), ("관광지", None), ("식당", "저녁")]
_MIDDLE = [("관광지", None), ("식당", "점심"), ("카페", None), ("관광지", None), ("식당", "저녁")]
_LAST = [("관광지", None)]


def build_slots(days: int) -> list[Slot]:
    slots = []
    for d in range(1, days + 1):
        template = _FIRST if d == 1 else _LAST if d == days else _MIDDLE
        for i, (cat, meal) in enumerate(template, start=1):
            slots.append(Slot(slot_id=f"d{d}-s{i}", day=d, order=i, category=cat, meal=meal))
    return slots


def assign_places(slots: list[Slot], candidates: dict[str, list[Place]]) -> dict[str, Place | None]:
    """슬롯 순서대로 카테고리 후보의 앞에서부터 배정. 한 일정 안에서 같은 장소는 한 번만."""
    used, out = set(), {}
    for s in slots:
        pick = next((p for p in candidates.get(s.category, []) if (p.name, p.region) not in used), None)
        if pick:
            used.add((pick.name, pick.region))
        out[s.slot_id] = pick
    return out


STAY_MIN = {"식당": 90, "카페": 60, "관광지": 120}


def assign_times(items: list[ItineraryItem], legs: dict[str, tuple[int, str | None]],
                 default_move: int = 30) -> list[ItineraryItem]:
    """일차별 시작(1일차 12:00, 이후 10:00)부터 체류·이동 시간을 누적. legs: slot_id → (직전 장소에서 이동 분, 교통편)."""
    out, cursor, day, prev_placed = [], None, None, False
    for it in items:
        if it.slot.day != day:
            day, prev_placed = it.slot.day, False
            cursor = datetime(2000, 1, 1, 12 if day == 1 else 10, 0)
        if it.place is None:
            out.append(it)
            continue
        update = {}
        if prev_placed:
            mv, detail = legs.get(it.slot.slot_id, (default_move, None))
            cursor += timedelta(minutes=mv)
            update.update(move_min=mv, move_detail=detail)
        end = cursor + timedelta(minutes=STAY_MIN.get(it.slot.category, 90))
        update.update(start=cursor.strftime("%H:%M"), end=end.strftime("%H:%M"))
        out.append(it.model_copy(update=update))
        cursor, prev_placed = end, True
    return out
