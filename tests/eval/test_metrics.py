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
