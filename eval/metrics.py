"""평가 지표. 모두 순수 함수 — LLM·I/O 없음."""
from collections import defaultdict

from eval.regions import region_match

FIRST = ["식당", "카페", "관광지", "식당"]
MIDDLE = ["관광지", "식당", "카페", "관광지", "식당"]
LAST = ["관광지"]


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def retrieval_item_metrics(cands: list[dict], item: dict, k: int) -> dict:
    top = cands[:k]
    if not top:
        return {"empty": 1, "returned": 0, "region_acc": 0.0, "region_hit1": 0,
                "category_acc": 0.0, "diversity": 0.0}
    reg = [region_match(c["region"], item["gold_sido"], item["gold_sigungu"]) for c in top]
    cat = [c["category"] == item["gold_category"] for c in top]
    return {"empty": 0, "returned": len(top), "region_acc": sum(reg) / len(top),
            "region_hit1": int(reg[0]), "category_acc": sum(cat) / len(top),
            "diversity": len({(c["name"], c["region"]) for c in top}) / len(top)}


def category_group(t: str) -> str:
    t = str(t)
    if any(x in t for x in ["식당", "맛집", "음식", "요리", "레스토랑"]):
        return "식당"
    if any(x in t for x in ["카페", "커피", "베이커리", "디저트"]):
        return "카페"
    return "관광지"


def expected_slots(days: int) -> list[list[str]]:
    # 기존 graph_flow 규칙: 1일차 4곳(점심 시작), 중간 날 5곳, 마지막 날 1곳. 1일 여행은 1일차 규칙.
    return [FIRST if d == 1 else LAST if d == days else MIDDLE for d in range(1, days + 1)]


def e2e_trip_metrics(itinerary: list[dict], trip: dict) -> dict:
    places = [x for x in itinerary if x.get("type") != "move"]
    moves = [x for x in itinerary if x.get("type") == "move"]
    exp = expected_slots(trip["days"])
    by_day = defaultdict(list)
    for x in places:
        by_day[int(x.get("day", 1))].append(category_group(x.get("type", "")))
    total = sum(len(e) for e in exp)
    filled = adhered = overflow = 0
    for d, e in enumerate(exp, start=1):
        actual = by_day.get(d, [])
        filled += min(len(actual), len(e))
        adhered += sum(a == b for a, b in zip(actual, e))
        overflow += max(0, len(actual) - len(e))
    overflow += sum(len(v) for d, v in by_day.items() if d > len(exp))
    names = [x.get("name") for x in places]
    return {
        "n_places": len(places),
        "slot_fill": filled / total,
        "category_adherence": adhered / total,
        "overflow": overflow,
        "region_acc": mean_or_none([float(region_match(x.get("address", ""), trip["gold_sido"],
                                                       trip["gold_sigungu"])) for x in places]),
        "dup_rate": (1 - len(set(names)) / len(names)) if names else 0.0,
        "total_move_min": sum(int(m.get("duration_min", 0)) for m in moves),
    }


def weighted_kappa(a: list[int], b: list[int], labels=(1, 2, 3, 4, 5)) -> float:
    n, k = len(a), len(labels)
    pos = {l: i for i, l in enumerate(labels)}
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[pos[x]][pos[y]] += 1
    ra = [sum(r) for r in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = lambda i, j: (i - j) ** 2 / (k - 1) ** 2
    num = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * ra[i] * cb[j] / n for i in range(k) for j in range(k))
    return 1.0 - num / den if den else 1.0


def aggregate(rows: list[dict], keys: list[str], by: str | None = None) -> dict:
    groups = {"all": rows}
    if by:
        for r in rows:
            groups.setdefault(r[by], []).append(r)
    return {g: {"n": len(rs), **{k: mean_or_none([r.get(k) for r in rs]) for k in keys}}
            for g, rs in groups.items()}
