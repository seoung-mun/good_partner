"""장소 단위 tool: 최적 장소 선택, 삭제, 교체."""
import json
from typing import List

from langchain_core.tools import tool

from src.tools.chains import desc_chain
from src.tools.retrieval import _extract_reviews, retrieve_candidates


@tool
async def find_and_select_best_place(query: str,
                                    destination: str,
                                    anchor: str = "",
                                    exclude_places: List[str] | None = None,
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

    reviews = _extract_reviews(best_doc)

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
async def delete_place(place_name: str) -> str:
    """일정에서 특정 장소를 삭제합니다."""
    return json.dumps({"action": "delete", "place_name": place_name}, ensure_ascii=False)

@tool
async def replace_place(old_place_name: str, query: str, destination: str) -> str:
    """일정 교체 도구"""
    return json.dumps({"action": "replace", "old": old_place_name, "new_query": query}, ensure_ascii=False)
