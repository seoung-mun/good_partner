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
RANK_K = 60              # 순위 점수 1/(K + 순위)의 K (RRF 표준값)


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


def _queries(category: str, keywords: list[str]) -> list[str]:
    """리뷰 말투 템플릿 쿼리. 지역은 선필터가 담당하므로 넣지 않는다. [취향 쿼리,] 기본 쿼리."""
    base = f"좋은 {QUERY_WORD[category]}"
    return ([f"{' '.join(keywords)} {base}"] if keywords else []) + [base]


def _quote(reviews: list[str], keywords: list[str]) -> str:
    """카드 인용: 키워드가 들어간 리뷰 우선, 없으면 가장 가까운 리뷰."""
    return next((r for r in reviews if any(k in r for k in keywords)), reviews[0] if reviews else "")


def search_candidates(db, region: str, categories: list[str], keywords: list[str] | dict[str, list[str]],
                      k: int = CHUNKS_PER_QUERY) -> dict[str, list[Place]]:
    """카테고리별 후보 장소(순위순). 모든 쿼리를 한 번에 배치 임베딩(CPU 서빙 기준 최적) — LLM 호출 없음.

    순위: 취향 쿼리 결과 먼저, 기본 쿼리에만 걸린 장소는 뒤에. score = 1/(60 + 순위) — 거리와 맞바꿀 때 쓰는 순위 점수.
    (RRF로 기본 쿼리·인기 순위를 동등하게 섞으면 취향 신호가 희석돼 실버 nDCG가 떨어짐 — docs/eval/p5.md)
    keywords: 모든 카테고리 공통 목록, 또는 카테고리별 dict(식당 키워드가 카페 검색에 섞이지 않게).
    k: 쿼리당 지역 안에서 가져올 리뷰 청크 수.
    """
    cat_ = get_catalog(db)
    kw_of = keywords if isinstance(keywords, dict) else {c: keywords for c in categories}
    plan = {c: _queries(c, kw_of.get(c, [])) for c in categories}
    texts = list(dict.fromkeys(q for qs in plan.values() for q in qs))
    vectors = dict(zip(texts, db.embedding_function.embed_documents(texts)))

    out: dict[str, list[Place]] = {}
    for c, queries in plan.items():
        ids = cat_.ids_for_region(region, c)
        evidence: dict[tuple[str, str], list[int]] = {}   # 삽입 순서 = 최종 순위
        for q in queries if len(ids) else []:
            for key, rows in _rank_places(lambda i: cat_.meta[i], *_region_search(db, ids, vectors[q], k)):
                evidence.setdefault(key, rows)
        places = []
        for rank, (key, rows) in enumerate(evidence.items(), start=1):
            st = cat_.places[key]
            reviews = [r for r in (cat_.meta[i].get("리뷰") for i in rows) if isinstance(r, str)]
            places.append(Place(name=key[0], region=key[1], category=c, address=st.address, rating=st.rating,
                                reviews=reviews, n_reviews=st.n_reviews, score=1 / (RANK_K + rank),
                                quote=_quote(reviews + st.reviews, kw_of.get(c, []))))
        out[c] = places
    return out
