"""장소 카탈로그: FAISS docstore를 한 번 훑어 지역별 행 id·장소별 통계(별점·리뷰 수·리뷰 원문)를 만든다.

지역 선필터 검색(IDSelector)과 UI 리뷰 원문 보기가 공유한다. 인덱스당 1회 구축 후 db 객체에 붙여 재사용.
"""
from dataclasses import dataclass, field

import numpy as np

from src.domain.region import region_matches


@dataclass
class PlaceStats:
    """장소 단위 통계 (리뷰 청크를 장소로 묶은 것)."""
    rating: float | None
    address: str
    category: str
    reviews: list[str] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)      # 이 장소의 리뷰 청크 faiss 행 id (시간대 적합도 계산용)

    @property
    def n_reviews(self) -> int:
        """수집된 리뷰 수."""
        return len(self.reviews)


class Catalog:
    """행 id → 메타데이터, 지역 → 행 id, (장소명, 지역) → PlaceStats."""

    def __init__(self, db):
        """docstore를 faiss 행 순서대로 훑어 구축."""
        self.meta: list[dict] = [db.docstore.search(db.index_to_docstore_id[i]).metadata
                                 for i in range(db.index.ntotal)]
        by_region: dict[str, list[int]] = {}
        self.places: dict[tuple[str, str], PlaceStats] = {}
        for i, md in enumerate(self.meta):
            region = md.get("지역", "")
            by_region.setdefault(region, []).append(i)
            key = (md.get("장소명", ""), region)
            st = self.places.get(key)
            if st is None:
                r = md.get("별점")
                st = self.places[key] = PlaceStats(rating=float(r) if r not in (None, "") and r == r else None,  # NaN 제외
                                                   address=str(md.get("상세 주소", "") or ""),
                                                   category=md.get("카테고리", ""))
            if isinstance(md.get("리뷰"), str) and md["리뷰"]:  # 결측 리뷰는 NaN(float)
                st.reviews.append(md["리뷰"])
                st.rows.append(i)
        self._by_region = {k: np.array(v, dtype="int64") for k, v in by_region.items()}
        self._category = np.array([md.get("카테고리", "") for md in self.meta])
        self._region_cache: dict[tuple, np.ndarray] = {}

    def ids_for_region(self, target: str, category: str | None = None) -> np.ndarray:
        """목표 지역(시도 [+ 시군구]) [및 카테고리] 안의 행 id. 고유 지역 문자열(~250개)에만 region_matches 적용."""
        key = (target, category)
        if key not in self._region_cache:
            parts = [ids for region, ids in self._by_region.items() if region_matches(region, target)]
            ids = np.concatenate(parts) if parts else np.array([], dtype="int64")
            self._region_cache[key] = ids[self._category[ids] == category] if category else ids
        return self._region_cache[key]


def get_catalog(db) -> Catalog:
    """db에 붙은 카탈로그를 반환 (없으면 구축). Streamlit·평가 모두 db 객체가 캐시되므로 인덱스당 1회."""
    cat = getattr(db, "_castone_catalog", None)
    if cat is None:
        cat = Catalog(db)
        db._castone_catalog = cat
    return cat
