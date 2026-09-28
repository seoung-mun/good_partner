"""일정 단위 tool: 타임라인 계산, 경로 순서 최적화, PDF 승인."""
import asyncio
import json
import traceback
from itertools import permutations
from typing import Dict, List

from langchain_core.tools import tool

from src.config import GMAPS_CLIENT


@tool
async def plan_itinerary_timeline(itinerary: List[Dict]) -> str:
    """
    [일정 정리 도구] 일정 리스트를 받아 시간순 타임라인 생성
    """
    print(f"\n--- plan_itinerary_timeline 호출 ---")
    places_only = [item for item in itinerary if item.get('type') != 'move']
    
    try:
        from src.scheduler.smart_scheduler import SmartScheduler
        scheduler = SmartScheduler(start_time_str="10:00")
        
        days = sorted(list(set(item.get('day', 1) for item in places_only)))
        final_timeline = []
        
        for day in days:
            day_items = [item for item in places_only if item.get('day', 1) == day]
            day_schedule = await scheduler.plan_day(day_items)
            
            for item in day_schedule:
                item['day'] = day
                if item.get('type') == 'move':
                    detail = item.get('transport_detail', '')
                    min_val = item.get('duration_min', 0)
                    item['duration_text'] = f"약 {min_val}분 ({detail})" if detail else f"약 {min_val}분 (이동)"
                final_timeline.append(item)
                
        return json.dumps(final_timeline, ensure_ascii=False)

    except Exception as e:
        print(f"ERROR: 스케줄링 실패: {e}")
        traceback.print_exc()
        return json.dumps(itinerary, ensure_ascii=False)

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

@tool
async def optimize_and_get_routes(places: List[str], start_location: str = "") -> str:
    """최적 경로(순서) 계산"""
    if not GMAPS_CLIENT: return "API 키 없음"
    all_places = [start_location] + places if start_location else places
    if len(all_places) < 2: return "장소 부족"

    try:
        matrix = await asyncio.to_thread(
            GMAPS_CLIENT.distance_matrix, origins=all_places, destinations=all_places, mode="transit"
        )
        dur_matrix = []
        for row in matrix['rows']:
            vals = [el.get('duration', {}).get('value', 99999) for el in row['elements']]
            dur_matrix.append(vals)
            
        best_indices, _ = await asyncio.to_thread(_solve_tsp, dur_matrix, bool(start_location), len(all_places))
        optimized = [all_places[i] for i in best_indices]
        
        return json.dumps({"optimized_order": optimized}, ensure_ascii=False)
        
    except Exception as e:
        return f"최적화 실패: {e}"


@tool
def confirm_and_download_pdf():
    """최종 승인 및 PDF 다운로드 활성화"""
    return "PDF 다운로드 승인됨"
