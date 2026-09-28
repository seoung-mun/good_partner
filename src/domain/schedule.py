"""일정 도메인 로직 (외부 의존 없음)."""
from itertools import permutations

from src.domain.models import Slot


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
