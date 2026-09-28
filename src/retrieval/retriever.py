"""장소 후보 검색: 지역 해석 → 멀티쿼리 벡터 검색 → 지역·카테고리 필터 → 2차 검색."""
import asyncio
from typing import Any, Dict, List

from src.domain.models import Place
from src.retrieval.store import load_faiss_index
from src.domain.region import canonical_region, region_matches
from src.agent.chains import query_gen_chain, region_chain
from src.services.maps import calculate_haversine_distance, get_coordinates, resolve_admin_region


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

def _extract_reviews(doc) -> List[str]:
    """추천 장소의 근거 리뷰. (기존 코드는 page_content를 줄 단위로 찾았으나 한 줄 포맷이라 항상 빈 리스트였음)"""
    md = doc.metadata
    if isinstance(md.get("reviews"), list):
        return [str(r) for r in md["reviews"][:3]]
    return [md["리뷰"]] if md.get("리뷰") else []


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
            key = (p_name, doc.metadata.get('지역', ''))  # 이름만으로 합치면 다른 지역 동명 가게가 사라짐
            if p_name and key not in seen_places and p_name not in exclude_places:
                seen_places.add(key)
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


QUERY_WORD = {"식당": "맛집", "카페": "카페", "관광지": "관광지"}


def _to_place(doc) -> Place:
    md = doc.metadata
    rating = md.get("별점")
    return Place(name=md.get("장소명", ""), region=md.get("지역", ""), category=md.get("카테고리", ""),
                 address=str(md.get("상세 주소", "") or ""), rating=float(rating) if rating not in (None, "") else None,
                 reviews=[md["리뷰"]] if md.get("리뷰") else [])


def search_candidates(db, region: str, categories: list[str], keywords: list[str], k: int = 50) -> dict[str, list[Place]]:
    """카테고리별 후보. 모든 쿼리를 한 번에 배치 임베딩(CPU 서빙 기준 최적) → 벡터 검색 → 지역·카테고리 필터.

    쿼리는 템플릿: 취향 쿼리('지역 키워드 카테고리')를 먼저, 기본 쿼리('지역 카테고리')를 뒤에 — LLM 호출 없음.
    """
    plan = []  # (category, query)
    for cat in categories:
        if keywords:
            plan.append((cat, f"{region} {' '.join(keywords)} {QUERY_WORD[cat]}"))
        plan.append((cat, f"{region} {QUERY_WORD[cat]}"))
    texts = list(dict.fromkeys(q for _, q in plan))
    vectors = dict(zip(texts, db.embedding_function.embed_documents(texts)))

    out: dict[str, list[Place]] = {c: [] for c in categories}
    index: dict[tuple, Place] = {}
    for cat, q in plan:
        for doc in db.similarity_search_by_vector(vectors[q], k=k):
            md = doc.metadata
            if md.get("카테고리") != cat or not region_matches(md.get("지역", ""), region):
                continue
            key = (md.get("장소명", ""), md.get("지역", ""))
            if key in index:  # 같은 장소의 다른 리뷰 chunk → 근거 리뷰만 추가
                if md.get("리뷰") and md["리뷰"] not in index[key].reviews and len(index[key].reviews) < 3:
                    index[key].reviews.append(md["리뷰"])
                continue
            index[key] = _to_place(doc)
            out[cat].append(index[key])
    return out
