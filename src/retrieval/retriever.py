"""장소 후보 검색: 카테고리별 쿼리를 한 번에 배치 임베딩 → 벡터 검색 → 지역·카테고리 필터 → 장소 단위 병합.

동기 함수(CPU 작업: 임베딩·FAISS). 비동기 호출부에서는 asyncio.to_thread로 감싼다.
"""
from src.domain.models import Place
from src.domain.region import region_matches

QUERY_WORD = {"식당": "맛집", "카페": "카페", "관광지": "관광지"}


def _to_place(doc) -> Place:
    """검색 문서(메타데이터) → Place."""
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
