import os, json, math, requests
import httpx
import asyncio
import datetime
import re 
from typing import List, Any, Dict, Optional, Tuple
import traceback
from itertools import permutations

from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.load import dumps, loads
from src.config import LLM, load_faiss_index, GMAPS_CLIENT

from src.region_cut_fuzz import canonical_region, normalize_region_name, region_matches


query_gen_prompt = PromptTemplate.from_template("""
역할: 당신은 '검색어 최적화 전문가'입니다.
목표: 사용자의 요청과 취향을 분석하여, 벡터 데이터베이스에서 가장 정확한 장소를 찾을 수 있는 **3개의 검색 쿼리**를 생성하세요.

[입력 정보]
- 여행지/지역: {target_region}
- 사용자 검색어: {query}
- 사용자 취향/정보: {user_info}
- 카테고리 필터: {category_filter}

[지침]
1. 사용자의 자연어 문장(취향)에서 **핵심 키워드(형용사, 명사)**만 추출하세요. (예: "조용한", "뷰맛집", "재즈")
2. 지역명과 핵심 키워드를 조합하여 검색어를 만드세요.
3. 다음 3가지 관점의 쿼리를 생성하세요:
   - 쿼리 1: 지역명 + 사용자 검색어 (기본 정확도 중심)
   - 쿼리 2: 지역명 + 사용자 검색어 + 취향 키워드 (구체적 니즈 중심)
   - 쿼리 3: 지역명 + 분위기/테마 키워드 (광범위 탐색)
4. 결과는 오직 쉼표(,)로 구분된 문자열로만 출력하세요. 다른 설명은 생략하세요.

[예시]
입력: 지역="서울", 검색어="카페", 취향="조용하고 작업하기 좋은 곳", 필터="카페"
출력: 서울 카페, 서울 조용한 작업하기 좋은 카페, 서울 스터디 카페 분위기
""")

query_gen_chain = query_gen_prompt | LLM | StrOutputParser()


region_prompt = PromptTemplate.from_template("""
역할: 당신은 '지명 정규화 전문가'입니다.
목표: 사용자의 검색어("{query}")와 여행 목적지("{destination}")를 보고, 검색 대상이 되는 **정확한 행정구역 명칭** 하나만 출력하세요.

[규칙]
1. 검색어에 '해운대', '송도' 같은 구체적 지명이 있다면, 해당 지명의 **공식 행정구역명**을 찾으세요.
2. 검색어가 '맛집', '카페' 등 일반 명사뿐이라면, **여행 목적지("{destination}")**를 정규화해서 반환하세요.
3. **절대 추측하지 마세요.** 모르면 "{destination}"을 그대로 반환하세요.
4. 답변에는 군더더기 없이 **오직 지역명만** 출력하세요.

[예시]
- 입력: "해운대 맛집", 목적지: "부산" -> 출력: "부산광역시 해운대구"
- 입력: "성산일출봉", 목적지: "제주도" -> 출력: "제주특별자치도 서귀포시"
- 입력: "강남 점심", 목적지: "서울" -> 출력: "서울특별시 강남구"
- 입력: "맛집 추천", 목적지: "여수" -> 출력: "전라남도 여수시"
""")
region_chain = region_prompt | LLM | StrOutputParser()

desc_prompt = PromptTemplate.from_template("""
[상황]
사용자 정보: {user_info}
장소 이름: {place_name}
장소 특징/리뷰 요약: {place_data}

위 정보를 바탕으로, 이 장소가 **이 사용자에게 왜 좋은지** 매력적인 1~2줄의 추천사를 작성해줘.
- 반드시 한국어로 작성.
- 문장 끝은 '해요', '좋아요' 등으로 자연스럽게 마무리.
""")
desc_chain = desc_prompt | LLM | StrOutputParser()



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



async def _search_docs(query_str: str, k: int = 20):
    """Vector DB 검색 래퍼"""
    try:
        print(f" 벡터 DB 검색 시도: '{query_str}'")
        db= load_faiss_index()
        if db is None:
            print(" 벡터 DB 인스턴스 없음")
            return []
        return await asyncio.to_thread(db.similarity_search, query_str, k=k)
    except Exception as e:
        print(f" DB 검색 실패: {e}")
        return []

async def _filter_candidates(docs, target_region: str, exclude_places: List[str], category_filter: str):
    """
    메타데이터 필터링 (지역명 + 카테고리 + 제외 장소)
    """
    candidates = []
    print(f" _filter_candidates 진입 (문서 수: {len(docs)})")
    
    print(f" 필터 적용 - 지역:{canonical_region(target_region)} / 카테고리:{category_filter}")
    count=0
    for doc in docs:
        count+=1
        name = doc.metadata.get('장소명', '이름미상')
        address = doc.metadata.get('지역', '') or doc.metadata.get('road_address', '')
        doc_cat = doc.metadata.get('카테고리', '')
        if count%100==0:
            print(f" 필터링 진행 중 ({count}/{len(docs)})")

        if name in exclude_places: continue

        if category_filter == "식당" or category_filter == "맛집":
            if not any(x in doc_cat for x in ["식당", "맛집", "음식점"]): continue
        elif category_filter == "카페":
            if not any(x in doc_cat for x in ["카페", "커피"]): continue
        elif category_filter == "관광지":
            if not any(x in doc_cat for x in ["관광", "여행", "명소"]): continue

        # 기존: 접미사를 떼고 부분 문자열 비교 → '광주광역시'와 '경기도 광주시', 동명 구가 서로 통과했음
        if region_matches(address, target_region):
            candidates.append(doc)
            
    return candidates

async def retrieve_candidates(query: str, destination: str, category_filter: str = "",
                              anchor: str = "", exclude_places: List[str] | None = None,
                              user_info: str = "") -> Dict[str, Any]:
    """지역 해석 → 멀티쿼리 검색 → 필터 → (없으면) 2차 검색. find_and_select_best_place의 검색 단계."""
    exclude_places = exclude_places or []
    target_input = anchor if anchor else query
    target_region = ""
    
    try:
        print(f" 지역명 정규화(LLM) 시도: '{target_input}' (목적지: {destination})")
        target_region = await region_chain.ainvoke({"query": target_input, "destination": destination})
        target_region = target_region.strip()
        print(f" LLM 정규화 결과: '{target_region}'")
    except Exception as e:
        print(f" LLM 지역명 실패({e}) -> Google Maps Fallback")
        
    if not target_region or target_region == destination:
        if anchor:
            target_region = await resolve_admin_region(anchor, destination)
        else:
            t_inp = query if destination in query else f"{destination} {query}"
            target_region = await resolve_admin_region(t_inp, destination)
            
    target_region = target_region.strip()

    center_place = anchor if anchor else target_region
    center_lat, center_lng = None, None
    if center_place:
        print(f" 기준점 좌표 조회: '{center_place}'")
        center_lat, center_lng = await get_coordinates(center_place)

    try:
        generated_queries_str = await query_gen_chain.ainvoke({
            "target_region": target_region,
            "query": query,
            "user_info": user_info,
            "category_filter": category_filter
        })
        search_queries = [q.strip() for q in generated_queries_str.split(',') if q.strip()]
        print(f" 생성된 멀티 쿼리: {search_queries}")
        
    except Exception as e:
        print(f" 쿼리 생성 실패({e}) -> 기본 쿼리 사용")
        search_queries = [f"{target_region} {query} {category_filter}"]

    tasks = [_search_docs(q, k=50) for q in search_queries]
    results_list = await asyncio.gather(*tasks)
    
    seen_places = set()
    aggregated_docs = []
    
    for docs in results_list:
        for doc in docs:
            p_name = doc.metadata.get('장소명', '')
            if p_name and p_name not in seen_places and p_name not in exclude_places:
                seen_places.add(p_name)
                aggregated_docs.append(doc)
    print(f" 통합된 문서 수: {len(aggregated_docs)} -> 필터링 진입")
    
    candidates = await _filter_candidates(aggregated_docs, target_region, exclude_places, category_filter)
    
    if len(candidates) > 5:
        print(f" 후보군 {len(candidates)}개 -> 상위 5개로 제한")
        candidates = candidates[:5]

    fallback_used = False
    if not candidates:
        fallback_used = True
        print(f" 1차 검색 결과 없음 -> 2차 검색(선호 제외) 전환")
        
        search_query_v2 = f"{target_region} {query} {category_filter}"
        print(f" 2차 검색 시도: '{search_query_v2}'")
        
        docs_v2 = await _search_docs(search_query_v2, k=30)
        candidates = await _filter_candidates(docs_v2, target_region, exclude_places, category_filter)
        
        if candidates and center_lat and center_lng:
            print(" 후보군 거리 정렬 시작")
            top_n_candidates = candidates[:5]
            candidates_with_score = []
            
            for doc in top_n_candidates:
                addr = doc.metadata.get('지역', '') or doc.metadata.get('road_address', '') or doc.metadata.get('상세 주소', '')
                p_lat, p_lng = await get_coordinates(addr)
                
                dist = 9999.0
                if p_lat and p_lng:
                    dist = calculate_haversine_distance(center_lat, center_lng, p_lat, p_lng)
                candidates_with_score.append((dist, doc))
            
            candidates_with_score.sort(key=lambda x: x[0])
            candidates = [x[1] for x in candidates_with_score]

    return {"target_region": target_region, "queries": search_queries,
            "candidates": candidates, "fallback_used": fallback_used}


@tool
async def find_and_select_best_place(query: str,
                                    destination: str,
                                    anchor: str = "",
                                    exclude_places: List[str] = [],
                                    user_info: str = "", 
                                    category_filter: str = "") -> str:
    """
    [핵심 도구] 최적의 장소 1곳을 반환합니다.
    [수정] Google Maps 좌표 오류(문경시 문제) 방지를 위해 LLM 지역명 정규화를 우선 사용
    """
    print(f"\n--- find_and_select_best_place 호출 ---")
    
 
    try:
        r = await retrieve_candidates(query, destination, category_filter, anchor, exclude_places, user_info)
    except Exception as e:
        print(f" 필터링 함수 내부 에러: {e}")
        return json.dumps({"name": "시스템에러", "type": "에러", "description": "필터링 중 에러 발생"}, ensure_ascii=False)
    candidates = r["candidates"]

    if not candidates:
        return json.dumps({"name": "추천 장소 없음", "type": "정보없음", "description": "조건에 맞는 장소를 찾지 못했습니다.", "reviews": []}, ensure_ascii=False)

    best_doc = candidates[0]
    best_name = best_doc.metadata.get('장소명', '이름미상')
    best_address = best_doc.metadata.get('지역', '')

    description = await desc_chain.ainvoke({
        "user_info": user_info,
        "place_name": best_name,
        "place_data": best_doc.page_content[:400]
    })

    reviews = []
    try:
        if 'reviews' in best_doc.metadata:
            reviews_data = best_doc.metadata.get('reviews', [])
            if isinstance(reviews_data, list):
                reviews = reviews_data[:3]
            elif isinstance(reviews_data, str):
                reviews = [r.strip() for r in reviews_data.split('\n') if r.strip()][:3]
        
        if not reviews and best_doc.page_content:
            content = best_doc.page_content
            if '리뷰' in content or 'review' in content.lower():
                lines = content.split('\n')
                review_start = False
                temp_reviews = []
                for line in lines:
                    if '리뷰' in line or 'review' in line.lower():
                        review_start = True
                        continue
                    if review_start and line.strip():
                        temp_reviews.append(line.strip())
                        if len(temp_reviews) >= 2: break
                reviews = temp_reviews
    except Exception as e:
        reviews = []

    result_data = {
        "name": best_name,
        "type": best_doc.metadata.get('카테고리', '장소명'), 
        "description": description.strip(),
        "address": best_address,
        "reviews": reviews,
        "coordinates": None 
    }
    
    print(f" 최종 추천: {best_name}")
    return json.dumps(result_data, ensure_ascii=False)



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
def get_weather_forecast(destination: str, dates: str) -> str:
    """
    도시명(destination)으로 위도/경도를 조회하고, 그 좌표로 5일 예보를 조회하여,
    사용자가 요청한 날짜(dates)의 날씨만 요약해 반환합니다. (3단계 날짜 파싱 적용)
    """
    API_KEY = os.getenv("OWM_API_KEY")
    if not API_KEY:
        return "오류: OWM_API_KEY가 .env 파일에 설정되지 않았습니다."

    geo_url = "https://api.openweathermap.org/geo/1.0/direct"
    geo_params = {'q': f"{destination},KR", 'limit': 1, 'appid': API_KEY}
    lat, lon = None, None
    try:
        response = requests.get(geo_url, params=geo_params, timeout=5)
        response.raise_for_status()
        geo_data = response.json()
        if geo_data:
            lat = geo_data[0]['lat']
            lon = geo_data[0]['lon']
        else:
            return f"오류: '{destination}'의 좌표(Geocoding)를 찾을 수 없습니다."
    except Exception as e:
        return f"오류: Geocoding API 호출 중 문제 발생: {e}"

    forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
    forecast_params = {'lat': lat, 'lon': lon, 'appid': API_KEY, 'units': 'metric', 'lang': 'kr'}
    forecasts = None
    try:
        response = requests.get(forecast_url, params=forecast_params, timeout=10)
        response.raise_for_status()
        forecast_data = response.json()
        forecasts = forecast_data.get('list', [])
    except Exception as e:
        return f"오류: Forecast API 호출 중 문제 발생: {e}"
    if not forecasts:
        return "오류: Forecast API에서 'list' 데이터를 찾을 수 없습니다."

    target_date_str = ""
    today = datetime.datetime.now()
    
    try:
        target_date_obj = datetime.datetime.strptime(dates, "%Y년 %m월 %d일")
        target_date_str = target_date_obj.strftime("%Y-%m-%d")
    except ValueError:
        try:
            target_date_obj = datetime.datetime.strptime(dates, "%Y년%m월%d일")
            target_date_str = target_date_obj.strftime("%Y-%m-%d")
        except ValueError:
            try:
                target_date_obj = datetime.datetime.strptime(dates, "%m월 %d일")
                target_date_obj = target_date_obj.replace(year=today.year)
                target_date_str = target_date_obj.strftime("%Y-%m-%d")
            except ValueError:
                 if "주말" in dates or "토요일" in dates:
                     days_until_saturday = (5 - today.weekday() + 7) % 7
                     saturday = today + datetime.timedelta(days=days_until_saturday)
                     target_date_str = saturday.strftime("%Y-%m-%d")
                 elif "내일" in dates:
                     tomorrow = today + datetime.timedelta(days=1)
                     target_date_str = tomorrow.strftime("%Y-%m-%d")
                 else: 
                     tomorrow = today + datetime.timedelta(days=1)
                     target_date_str = tomorrow.strftime("%Y-%m-%d")
    
    output_str = f"[{destination} ({target_date_str}) 날씨 예보 (OWM)]\n"
    found = False
    for forecast in forecasts:
        if forecast['dt_txt'].startswith(target_date_str):
            time_utc = forecast['dt_txt'].split(' ')[1][:5]
            temp = forecast['main']['temp'] 
            desc = forecast['weather'][0]['description']
            output_str += f"- {time_utc} (UTC): {temp:.1f}℃, {desc}\n"
            found = True
    
    if not found:
        return f"정보: {target_date_str} 날짜의 예보를 찾을 수 없습니다. (OWM은 5일치만 제공)"
    
    return output_str


@tool
def confirm_and_download_pdf():
    """최종 승인 및 PDF 다운로드 활성화"""
    return "PDF 다운로드 승인됨"

@tool
async def delete_place(place_name: str) -> str:
    """일정에서 특정 장소를 삭제합니다."""
    return json.dumps({"action": "delete", "place_name": place_name}, ensure_ascii=False)

@tool
async def replace_place(old_place_name: str, query: str, destination: str) -> str:
    """일정 교체 도구"""
    return json.dumps({"action": "replace", "old": old_place_name, "new_query": query}, ensure_ascii=False)


TOOLS = [
    find_and_select_best_place,
    plan_itinerary_timeline,
    optimize_and_get_routes,
    get_weather_forecast,
    delete_place,
    replace_place,
    confirm_and_download_pdf
]
AVAILABLE_TOOLS = {tool.name: tool for tool in TOOLS}