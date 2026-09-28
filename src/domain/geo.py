"""순수 거리·시간 계산 (외부 의존 없음)."""
import math


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """두 좌표 간의 직선 거리(km) 계산 (Pure Python)"""
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except (ValueError, TypeError):
        return 9999.0

    R = 6371   
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_distance_time(start_lat, start_lng, end_lat, end_lng, mode="driving"):
    """좌표 간 단순 직선 거리 및 예상 시간 추정"""
    dist = calculate_haversine_distance(start_lat, start_lng, end_lat, end_lng)
    
    speed = 4.0 if mode == "walking" else 30.0
    seconds = int((dist / speed) * 3600)
    
    if seconds < 3600: text = f"{seconds // 60}분"
    else: text = f"{seconds // 3600}시간 {(seconds % 3600) // 60}분"
    return dist, seconds, text
