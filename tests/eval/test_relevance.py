from eval.build_golden import build_pref_items
from eval.run_relevance import relevance_summary

TRIPS = [{"id": "t01", "destination": "광주", "start_location": "광주송정역", "days": 2, "group_type": "친구",
          "style": "맛집 탐방", "detail": "떡갈비", "gold_sido": "광주광역시", "gold_sigungu": None}]


def test_pref_items_carry_preference_and_trip_fields():
    items = build_pref_items(TRIPS)
    assert [i["category_filter"] for i in items] == ["식당", "카페", "관광지"]
    it = items[0]
    assert it["id"] == "pref-t01-식당" and it["group"] == "pref" and it["query"] == "맛집"
    assert "상세 요청: 떡갈비" in it["user_info"]
    assert it["detail"] == "떡갈비" and it["gold_sido"] == "광주광역시" and it["gold_category"] == "식당"


def test_relevance_summary():
    rows = [{"scores": [5, 4, None]}, {"scores": [2]}, {"scores": []}]
    s = relevance_summary(rows)
    assert s["n_scored_items"] == 2
    assert s["mean_relevance"] == (4.5 + 2) / 2          # 항목별 평균의 평균
    assert s["relevant_rate"] == 2 / 3                    # 채점된 장소 중 4점 이상
