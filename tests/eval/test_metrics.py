import pytest
from eval.metrics import (aggregate, category_group, e2e_trip_metrics, expected_slots,
                          mean_or_none, retrieval_item_metrics, weighted_kappa)

ITEM = {"gold_sido": "부산광역시", "gold_sigungu": "중구", "gold_category": "식당"}


def c(name, region, cat="식당"):
    return {"name": name, "region": region, "category": cat}


def test_retrieval_metrics_basic():
    cands = [c("A", "부산광역시 중구"), c("B", "대구광역시 중구"), c("C", "부산광역시 중구", "카페")]
    m = retrieval_item_metrics(cands, ITEM, k=5)
    assert m["returned"] == 3 and m["empty"] == 0
    assert m["region_acc"] == pytest.approx(2 / 3)
    assert m["region_hit1"] == 1
    assert m["category_acc"] == pytest.approx(2 / 3)
    assert m["diversity"] == 1.0


def test_retrieval_metrics_empty():
    m = retrieval_item_metrics([], ITEM, k=5)
    assert m == {"empty": 1, "returned": 0, "region_acc": 0.0, "region_hit1": 0,
                 "category_acc": 0.0, "diversity": 0.0}


def test_retrieval_diversity_counts_same_place_once():
    m = retrieval_item_metrics([c("A", "부산광역시 중구")] * 2, ITEM, k=5)
    assert m["diversity"] == 0.5


def test_expected_slots():
    assert expected_slots(1) == [["식당", "카페", "관광지", "식당"]]
    assert expected_slots(3) == [["식당", "카페", "관광지", "식당"],
                                 ["관광지", "식당", "카페", "관광지", "식당"], ["관광지"]]


def test_category_group():
    assert category_group("맛집") == "식당"
    assert category_group("커피전문점") == "카페"
    assert category_group("관광지") == "관광지"


def p(name, t, day, addr="부산광역시 중구"):
    return {"name": name, "type": t, "day": day, "address": addr}


def test_e2e_metrics():
    trip = {"days": 2, "gold_sido": "부산광역시", "gold_sigungu": "중구"}
    itin = [p("a", "식당", 1), {"type": "move", "duration_min": 10, "day": 1}, p("b", "카페", 1),
            p("c", "식당", 1), p("d", "식당", 1, "대구광역시 중구"),
            p("e", "관광지", 2), p("e", "관광지", 2)]
    m = e2e_trip_metrics(itin, trip)
    assert m["n_places"] == 6
    assert m["slot_fill"] == pytest.approx(5 / 5)          # day1 4/4, day2 1/1
    assert m["category_adherence"] == pytest.approx(4 / 5)  # day1 3번째 식당≠관광지
    assert m["overflow"] == 1                               # day2 1곳 초과
    assert m["region_acc"] == pytest.approx(5 / 6)
    assert m["dup_rate"] == pytest.approx(1 - 5 / 6)
    assert m["total_move_min"] == 10


def test_e2e_metrics_empty_itinerary():
    m = e2e_trip_metrics([], {"days": 1, "gold_sido": "부산광역시", "gold_sigungu": None})
    assert m["n_places"] == 0 and m["slot_fill"] == 0 and m["region_acc"] is None


def test_weighted_kappa():
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert weighted_kappa([1, 1, 5, 5], [5, 5, 1, 1]) < 0


def test_aggregate_ignores_none_and_groups():
    rows = [{"g": "x", "v": 1.0}, {"g": "x", "v": None}, {"g": "y", "v": 0.0}]
    assert mean_or_none([None, None]) is None
    out = aggregate(rows, ["v"], by="g")
    assert out["all"]["v"] == 0.5 and out["x"]["v"] == 1.0 and out["y"]["n"] == 1


def test_trip_outcome():
    from eval.metrics import trip_outcome
    assert trip_outcome({"ok": 0.0, "n_places": 0}) == "error"
    assert trip_outcome({"ok": 1.0, "n_places": 0}) == "empty"     # 예외 없이 끝났지만 일정 없음(도구 없이 답변 등)
    assert trip_outcome({"ok": 1.0, "n_places": 3}) == "success"


def test_summarize_e2e_uses_success_only_for_quality():
    from eval.metrics import summarize_e2e
    rows = [{"ok": 1.0, "n_places": 4, "slot_fill": 1.0, "dup_rate": 0.2, "llm_calls": 30},
            {"ok": 1.0, "n_places": 0, "slot_fill": 0.0, "dup_rate": 0.0, "llm_calls": 3},
            {"ok": 0.0, "n_places": 0, "slot_fill": 0.0, "dup_rate": 0.0, "llm_calls": 50}]
    s = summarize_e2e(rows, ["slot_fill", "dup_rate", "llm_calls"])
    assert s["n"] == 3 and s["n_success"] == 1
    assert s["success_rate"] == 1 / 3 and s["empty_rate"] == 1 / 3 and s["error_rate"] == 1 / 3
    assert s["slot_fill"] == 1.0 and s["dup_rate"] == 0.2 and s["llm_calls"] == 30
    assert s["llm_calls_all"] == (30 + 3 + 50) / 3                  # 비용은 실패 포함 전체 평균도 함께


def test_move_time_is_none_without_timeline():
    trip = {"days": 1, "gold_sido": "부산광역시", "gold_sigungu": None}
    no_tl = e2e_trip_metrics([p("a", "식당", 1), p("b", "카페", 1)], trip)
    assert no_tl["has_timeline"] == 0 and no_tl["total_move_min"] is None
    tl = e2e_trip_metrics([p("a", "식당", 1), {"type": "move", "duration_min": 32, "day": 1}, p("b", "카페", 1)], trip)
    assert tl["has_timeline"] == 1 and tl["total_move_min"] == 32


def test_summarize_e2e_service_metrics_over_all_trips():
    from eval.metrics import summarize_e2e
    rows = [{"ok": 1.0, "n_places": 2, "latency_s": 10, "tool_error_rate": 0.0},
            {"ok": 0.0, "n_places": 0, "latency_s": 100, "tool_error_rate": 0.5}]
    s = summarize_e2e(rows, [], all_keys=["tool_error_rate"])
    assert s["tool_error_rate_all"] == 0.25
    assert s["latency_p50_all"] == 10 and s["latency_p95_all"] == 100


def test_node_time_mean():
    from eval.run_e2e import node_time_mean
    rows = [{"node_time_s": {"a": 1.0, "b": 2.0}}, {"node_time_s": {"a": 3.0}}]
    assert node_time_mean(rows) == {"a": 2.0, "b": 1.0}
