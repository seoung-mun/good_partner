"""A층(검색 컴포넌트) 골든셋 생성. 정답은 메타데이터의 지역·카테고리 → 자동 채점."""
import json
import random
from collections import defaultdict
from pathlib import Path

from eval.metadata import place_counts
from eval.regions import SIDO_SHORT

CATS = ["식당", "카페", "관광지"]
QUERY = {"식당": "맛집", "카페": "카페", "관광지": "관광지"}
MIN_PLACES = 5
OUT = Path("eval/data/golden_retrieval.jsonl")

# 목적지는 광역 단위, 쿼리에 지명이 들어가는 경우 (앱은 목적지 기준으로 지역을 해석하므로 약점 측정용)
LANDMARKS = [
    ("부산", "해운대 맛집", "식당", "부산광역시", "해운대구"),
    ("부산", "광안리 카페", "카페", "부산광역시", "수영구"),
    ("부산", "서면 맛집", "식당", "부산광역시", "부산진구"),
    ("서울", "홍대 카페", "카페", "서울특별시", "마포구"),
    ("서울", "강남역 맛집", "식당", "서울특별시", "강남구"),
    ("전북", "전주 한옥마을 관광지", "관광지", "전북특별자치도", "전주시"),
    ("경북", "경주 황리단길 카페", "카페", "경상북도", "경주시"),
    ("강원", "속초 중앙시장 맛집", "식당", "강원특별자치도", "속초시"),
]


def _item(group, dest, query, cat, sido, sigungu):
    return {"id": f"{group}-{sido}-{sigungu or '전체'}-{cat}-{query}", "group": group,
            "destination": dest, "query": query, "category_filter": cat,
            "gold_sido": sido, "gold_sigungu": sigungu, "gold_category": cat}


def build_items(seed: int = 42, n_regular: int = 30) -> list[dict]:
    counts = place_counts()
    ok = lambda s, g, c: counts.get((s, g, c), 0) >= MIN_PLACES
    regions = sorted({(s, g) for (s, g, _c) in counts if g})
    items: list[dict] = []

    # 1) 이름이 같은 시군구 (중구·동구·고성군 …)
    by_name = defaultdict(list)
    for s, g in regions:
        by_name[g].append(s)
    amb = [(s, g) for g, sidos in sorted(by_name.items()) if len(sidos) > 1 for s in sorted(sidos)]
    for n, (s, g) in enumerate(amb):
        cat = next((c for c in CATS[n % 3:] + CATS[:n % 3] if ok(s, g, c)), None)
        if cat:
            items.append(_item("ambiguous", f"{SIDO_SHORT[s]} {g}", QUERY[cat], cat, s, g))

    # 2) 광주광역시 vs 경기도 광주시
    for cat in CATS:
        items.append(_item("gwangju", "광주", QUERY[cat], cat, "광주광역시", None))
        items.append(_item("gwangju", "경기 광주", QUERY[cat], cat, "경기도", "광주시"))

    # 3) 일반 시군구 무작위
    amb_set = set(amb) | {("경기도", "광주시")}
    pool = [(s, g) for s, g in regions if (s, g) not in amb_set and not g.endswith("구")]
    rng = random.Random(seed)
    for n, (s, g) in enumerate(rng.sample(pool, min(n_regular, len(pool)))):
        cat = CATS[n % 3]
        if ok(s, g, cat):
            items.append(_item("regular", f"{SIDO_SHORT[s]} {g}", QUERY[cat], cat, s, g))

    # 4) 지명 쿼리
    for dest, query, cat, s, g in LANDMARKS:
        if ok(s, g, cat):
            items.append(_item("landmark", dest, query, cat, s, g))
    return items


def build_pref_items(trips: list[dict]) -> list[dict]:
    """취향 쿼리 세트: B층 여행 요청의 상세 취향 × 카테고리. 관련도(LLM-as-judge) 측정용."""
    from eval.legacy_adapter import build_preference
    items = []
    for t in trips:
        for cat in CATS:
            items.append({**t, "id": f"pref-{t['id']}-{cat}", "group": "pref", "query": QUERY[cat],
                          "category_filter": cat, "gold_category": cat, "user_info": build_preference(t)})
    return items


# --- 실버 라벨 검색 골든셋 (MRR/nDCG용) --------------------------------------------
# 정답 = 리뷰에 속성어(동의어)가 min_hits회 이상 나온 장소. LLM 없이 메타데이터만 — 근사치이므로 전후 비교용.
# 동의어는 오탐을 피하도록 고름: '뷰'(리뷰), '아이'(아이스), '회'(회사) 같은 부분 문자열은 쓰지 않는다.
ATTRS = {
    "식당": {"해산물": ["해산물", "해물", "활어", "물회"], "아이랑": ["아이들", "아이랑", "아기", "키즈"],
             "주차 편한": ["주차"], "고기": ["고기", "삼겹살", "갈비"], "혼밥": ["혼밥"], "국밥": ["국밥"]},
    "카페": {"디저트": ["디저트", "케이크", "마카롱"], "오션뷰": ["오션뷰", "바다뷰", "바다가 보"],
             "조용한": ["조용"], "빵": ["빵", "베이커리"], "주차 편한": ["주차"], "넓은": ["넓", "대형"]},
    "관광지": {"야경": ["야경"], "산책": ["산책", "걷기"], "아이랑": ["아이들", "아이랑", "키즈"],
               "사진": ["사진", "포토"], "바다": ["바다", "해변"], "역사": ["역사", "문화재", "유적"]},
}
SEARCH_REGIONS = ["부산광역시", "서울특별시", "제주특별자치도", "강원특별자치도", "경상북도 경주시",
                  "전북특별자치도 전주시", "광주광역시", "대구광역시", "인천광역시", "전라남도 여수시"]
SEARCH_OUT = Path("eval/data/golden_search.jsonl")


def relevant_places(rows: list[dict], region: str, cat: str, synonyms: list[str], min_hits: int = 2) -> set:
    """지역·카테고리 안에서 리뷰에 동의어가 min_hits회 이상 나온 (장소명, 지역) 집합."""
    from src.domain.region import region_matches
    hits: dict = defaultdict(int)
    for r in rows:
        if r["카테고리"] == cat and isinstance(r.get("리뷰"), str) and any(w in r["리뷰"] for w in synonyms) and region_matches(r["지역"], region):
            hits[(r["장소명"], r["지역"])] += 1
    return {k for k, n in hits.items() if n >= min_hits}


def build_search_items(min_rel: int = 5, max_rel: int = 200) -> list[dict]:
    """지역 × 카테고리 × 속성어. 정답 수가 [min_rel, max_rel]인 항목만."""
    from eval.metadata import load_metadata
    rows = load_metadata()
    items = []
    for region in SEARCH_REGIONS:
        for cat, attrs in ATTRS.items():
            for kw, syn in attrs.items():
                rel = relevant_places(rows, region, cat, syn)
                if min_rel <= len(rel) <= max_rel:
                    items.append({"id": f"search-{region}-{cat}-{kw}", "region": region, "category": cat,
                                  "keyword": kw, "relevant": sorted(map(list, rel))})
    return items


def main():
    trips = [json.loads(l) for l in Path("eval/data/golden_trips.jsonl").read_text().splitlines() if l.strip()]
    pref = build_pref_items(trips)
    Path("eval/data/golden_pref.jsonl").write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in pref) + "\n")
    print(f"{len(pref)} pref items → eval/data/golden_pref.jsonl")
    items = build_items()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n")
    search = build_search_items()
    SEARCH_OUT.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in search) + "\n")
    print(f"{len(search)} search items → {SEARCH_OUT}")
    by = defaultdict(int)
    for i in items:
        by[i["group"]] += 1
    print(f"{len(items)} items → {OUT}  {dict(by)}")


if __name__ == "__main__":
    main()
