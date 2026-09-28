"""장소 후보 검색: 카테고리별 쿼리를 한 번에 배치 임베딩 → 지역·카테고리 안에서만 벡터 검색 → 장소 단위 점수.

동기 함수(CPU 작업: 임베딩·FAISS). 비동기 호출부에서는 asyncio.to_thread로 감싼다.
"""
import faiss
import numpy as np

from src.domain.models import Place
from src.retrieval.catalog import get_catalog

QUERY_WORD = {"식당": "맛집", "카페": "카페", "관광지": "관광지"}
CHUNKS_PER_QUERY = 300   # 지역 안에서 가져올 리뷰 청크 수 → 장소로 묶는다
TOP_CHUNKS = 3           # 장소 점수 = 가까운 리뷰 3개의 평균 거리


def _region_search(db, ids: np.ndarray, vec, k: int) -> tuple[np.ndarray, np.ndarray]:
    """ids(지역·카테고리 행) 안에서만 최근접 k개. IndexFlat은 IDSelector 선필터를 지원한다."""
    params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids))
    D, I = db.index.search(np.asarray([vec], dtype="float32"), min(k, len(ids)), params=params)
    keep = I[0] >= 0
    return D[0][keep], I[0][keep]


def _rank_places(cat_meta, D, I) -> list[tuple[tuple[str, str], list[int]]]:
    """청크 결과 → 장소 순위. 리뷰 3개 미만으로 걸린 장소는 검색된 것 중 가장 먼 거리로 채워 한 줄 운빨을 누른다."""
    if len(D) == 0:
        return []
    by_place: dict[tuple[str, str], list[tuple[float, int]]] = {}
    for d, i in zip(D, I):
        md = cat_meta(i)
        by_place.setdefault((md.get("장소명", ""), md.get("지역", "")), []).append((float(d), int(i)))
    worst = float(D.max())
    scored = []
    for key, hits in by_place.items():
        hits = sorted(hits)[:TOP_CHUNKS]
        dists = [d for d, _ in hits] + [worst] * (TOP_CHUNKS - len(hits))
        scored.append((sum(dists) / TOP_CHUNKS, key, [i for _, i in hits]))
    scored.sort(key=lambda x: x[0])
    return [(key, rows) for _, key, rows in scored]


def search_candidates(db, region: str, categories: list[str], keywords: list[str],
                      k: int = CHUNKS_PER_QUERY) -> dict[str, list[Place]]:
    """카테고리별 후보 장소(순위순). 모든 쿼리를 한 번에 배치 임베딩(CPU 서빙 기준 최적).

    쿼리는 템플릿: 취향 쿼리('지역 키워드 카테고리') 결과를 먼저, 기본 쿼리('지역 카테고리') 결과를 뒤에 — LLM 호출 없음.
    k: 쿼리당 지역 안에서 가져올 리뷰 청크 수.
    """
    cat_ = get_catalog(db)
    plan = []  # (category, query)
    for c in categories:
        if keywords:
            plan.append((c, f"{region} {' '.join(keywords)} {QUERY_WORD[c]}"))
        plan.append((c, f"{region} {QUERY_WORD[c]}"))
    texts = list(dict.fromkeys(q for _, q in plan))
    vectors = dict(zip(texts, db.embedding_function.embed_documents(texts)))

    out: dict[str, list[Place]] = {c: [] for c in categories}
    seen: set[tuple[str, str]] = set()
    for c, q in plan:
        ids = cat_.ids_for_region(region, c)
        if len(ids) == 0:
            continue
        D, I = _region_search(db, ids, vectors[q], k)
        for key, rows in _rank_places(lambda i: cat_.meta[i], D, I):
            if key in seen:
                continue
            seen.add(key)
            st = cat_.places[key]
            out[c].append(Place(name=key[0], region=key[1], category=c, address=st.address, rating=st.rating,
                                reviews=[r for r in (cat_.meta[i].get("리뷰") for i in rows) if isinstance(r, str)]))
    return out
