"""평가 지표. 모두 순수 함수 — LLM·I/O 없음."""
import math
from collections import defaultdict

from eval.regions import region_match
from eval.trace import percentile

# P5 템플릿 (P4 이전: 첫날 식당·카페·관광·식당, 중간 관광·식당·카페·관광·식당, 마지막 관광 1곳)
FIRST = ["식당", "관광지", "카페", "식당"]
MIDDLE = ["식당", "관광지", "식당", "관광지", "카페", "식당"]
LAST = ["식당", "관광지", "식당"]


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
        # 타임라인 계산(plan_itinerary_timeline)까지 간 일정만 이동 시간이 존재 — 없으면 0이 아니라 None
        "has_timeline": int(bool(moves)),
        "total_move_min": sum(int(m.get("duration_min", 0)) for m in moves) if moves else None,
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


def trip_outcome(row: dict) -> str:
    """error: 예외(무한 루프 등) / empty: 예외 없이 끝났지만 일정 없음(도구 없이 답변 포함) / success."""
    if not row.get("ok"):
        return "error"
    return "success" if row.get("n_places", 0) > 0 else "empty"


def summarize_e2e(rows: list[dict], quality_keys: list[str], all_keys: list[str] = ()) -> dict:
    """비율은 전체 기준, 품질·비용 지표는 성공한 여행만 평균. 비용은 실패 포함 전체 평균(*_all)도 함께."""
    outcomes = [trip_outcome(r) for r in rows]
    ok_rows = [r for r, o in zip(rows, outcomes) if o == "success"]
    rate = lambda o: outcomes.count(o) / len(rows) if rows else None
    cost_keys = [k for k in ("llm_calls", "input_tokens", "output_tokens", "latency_s") if k in quality_keys]
    return {"n": len(rows), "n_success": len(ok_rows), "success_rate": rate("success"),
            "empty_rate": rate("empty"), "error_rate": rate("error"),
            **{k: mean_or_none([r.get(k) for r in ok_rows]) for k in quality_keys},
            **{f"{k}_all": mean_or_none([r.get(k) for r in rows]) for k in [*cost_keys, *all_keys]},
            # 서비스 지연은 실패 포함 전체 요청 기준 백분위수
            "latency_p50_all": percentile([r["latency_s"] for r in rows if "latency_s" in r], 50),
            "latency_p95_all": percentile([r["latency_s"] for r in rows if "latency_s" in r], 95)}


def mrr_at_k(ranked: list, relevant: set, k: int) -> float:
    """상위 k 안의 첫 정답 순위의 역수 (없으면 0)."""
    return next((1 / (i + 1) for i, x in enumerate(ranked[:k]) if x in relevant), 0.0)


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    """상위 k가 정답 전체 중 몇 %를 포함하는지."""
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    """이진 관련도 nDCG@k."""
    dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(ranked[:k]) if x in relevant)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def _is_dinner(x: dict) -> bool:
    """저녁 식사 슬롯. meal 필드가 없는 옛 결과는 템플릿 위치(1일차 4번째, 이후 5번째)로 판별."""
    if "meal" in x:
        return x["meal"] == "저녁"
    sid = x.get("slot_id", "")
    return x.get("type") == "식당" and (sid.startswith("d1-") and sid.endswith("-s4")
                                       or not sid.startswith("d1-") and sid.endswith("-s5"))


def route_metrics(rows: list[dict]) -> dict:
    """동선 현실성 (성공 건만): 구간 이동 p50·p95, 60분 초과 구간 비율, 저녁 20시 이후 시작 비율, 여행당 총 이동(분)."""
    ok = [r for r in rows if r.get("ok")]
    legs = [int(x["duration_min"]) for r in ok for x in r["itinerary"] if x.get("type") == "move"]
    dinners = [x for r in ok for x in r["itinerary"] if x.get("type") != "move" and _is_dinner(x) and x.get("start")]
    return {"legs": len(legs), "leg_p50": percentile(legs, 50), "leg_p95": percentile(legs, 95),
            "leg_over60_rate": sum(m > 60 for m in legs) / len(legs) if legs else None,
            "late_dinner_rate": sum(x["start"] >= "20:00" for x in dinners) / len(dinners) if dinners else None,
            "move_per_trip": sum(legs) / len(ok) if ok else None}
