"""지오코딩·거리·경로·행정구역 해석 (Google Maps)."""
import asyncio
import datetime
import math

from src.config import GMAPS_CLIENT
from src.region_cut_fuzz import normalize_region_name


async def get_coordinates(location_name: str):
    """지명/주소 -> 좌표 변환 (Google Maps API)"""
    if not GMAPS_CLIENT: return None, None
    try:
        res = await asyncio.to_thread(GMAPS_CLIENT.geocode, location_name, language='ko')
        if res:
            loc = res[0]['geometry']['location']
            return loc['lat'], loc['lng']
    except Exception as e:
        print(f" 좌표 변환 실패 ({location_name}): {e}")
    return None, None

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

async def get_detailed_route(start_place: str, end_place: str, mode="transit", departure_time=None):
    """상세 경로 조회 (Google Maps Directions API)"""
    if not GMAPS_CLIENT: 
        print(f" GMAPS_CLIENT가 없습니다. (API Key 확인 필요)")
        return None
    if mode == "transit" and not departure_time: departure_time = datetime.datetime.now()
    if mode != "transit": departure_time = None

    try:
        print(f" 경로 검색 요청: {start_place} -> {end_place}")
        res = await asyncio.to_thread(
            GMAPS_CLIENT.directions, origin=start_place, destination=end_place,
            mode=mode, departure_time=departure_time, region="KR", language="ko"
        )
        if res:
            route = res[0]['legs'][0]
            steps_summary = []
            for step in route['steps']:
                tm = step['travel_mode']
                if tm == 'TRANSIT':
                    line = step.get('transit_details', {}).get('line', {})
                    name = line.get('short_name') or line.get('name') or "버스"
                    steps_summary.append(f"[{line.get('vehicle', {}).get('name', '대중교통')}] {name}")
                elif tm == 'WALKING': steps_summary.append("🚶 도보")
            
            if not steps_summary: steps_summary.append(f"이동 ({route['duration']['text']})")

            return {
                "mode": mode, "duration": route['duration']['text'],
                "duration_value": route['duration']['value'], "distance": route['distance']['text'],
                "steps": steps_summary,
                "start_location": route['start_location'], "end_location": route['end_location']
            }
    except Exception as e:
        print(f" 경로 검색 API 에러: {e}") 
        return None
    

    slat, slng = await get_coordinates(start_place)
    elat, elng = await get_coordinates(end_place)
    if slat and elat:
        dist, sec, txt = calculate_distance_time(slat, slng, elat, elng, mode)
        return {"mode": mode, "duration": txt, "duration_value": sec, "distance": f"{dist:.1f}km", "steps": ["직선거리"], "start_location": {"lat":slat, "lng":slng}, "end_location": {"lat":elat, "lng":elng}}
    return None

async def resolve_admin_region(query: str, destination: str) -> str:
    """
    [핵심 로직] "광안리" -> "부산광역시 수영구" 자동 변환기
    """
    if not GMAPS_CLIENT: 
        return normalize_region_name(destination)

    search_term = query
    if destination and destination not in query:
        search_term = f"{destination} {query}"
        
    print(f" 행정구역 식별 시도: '{search_term}'")

    try:
        geocode_res = await asyncio.to_thread(GMAPS_CLIENT.geocode, search_term, language='ko')
        
        if not geocode_res:
            return normalize_region_name(destination)

        loc = geocode_res[0]['geometry']['location']
        lat, lng = loc['lat'], loc['lng']
        
        reverse_res = await asyncio.to_thread(GMAPS_CLIENT.reverse_geocode, (lat, lng), language='ko')
        
        if not reverse_res:
            return normalize_region_name(destination)
            
        comps = reverse_res[0].get('address_components', [])
        level1 = "" 
        level2 = "" 
        
        for c in comps:
            types = c.get('types', [])
            if 'administrative_area_level_1' in types:
                level1 = c.get('long_name', '')
            elif 'sublocality_level_1' in types:
                level2 = c.get('long_name', '')
            elif 'locality' in types and not level2:
                level2 = c.get('long_name', '')
                
        extracted_region = f"{level1} {level2}".strip()
        
        if extracted_region:
            print(f" 변환 성공: '{query}' -> '{extracted_region}'")
            return extracted_region
        else:
            return normalize_region_name(destination)

    except Exception as e:
        print(f" 행정구역 변환 중 에러: {e}")
        return normalize_region_name(destination)
