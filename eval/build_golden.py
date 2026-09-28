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


def main():
    items = build_items()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n")
    by = defaultdict(int)
    for i in items:
        by[i["group"]] += 1
    print(f"{len(items)} items → {OUT}  {dict(by)}")


if __name__ == "__main__":
    main()
