import pickle
from collections import Counter, defaultdict
from functools import lru_cache

from eval.regions import canonical_region

META_PATH = "data/metadata_list.pkl"


@lru_cache(maxsize=1)
def load_metadata() -> list[dict]:
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def place_counts() -> dict[tuple[str, str | None, str], int]:
    seen = set()
    counts: Counter = Counter()
    for r in load_metadata():
        sido, sigungu = canonical_region(r["지역"])
        key = (sido, sigungu, r["카테고리"], r["장소명"])
        if key not in seen:
            seen.add(key)
            counts[(sido, sigungu, r["카테고리"])] += 1
    return dict(counts)


@lru_cache(maxsize=1)
def _reviews_index() -> dict[tuple[str, str, str | None], list[str]]:
    idx: dict = defaultdict(list)
    for r in load_metadata():
        idx[(r["장소명"], *canonical_region(r["지역"]))].append(r["리뷰"])
    return idx


def reviews_for(name: str, region: str, limit: int = 5) -> list[str]:
    return _reviews_index().get((name, *canonical_region(region)), [])[:limit]
