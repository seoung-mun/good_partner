from src.domain.models import Place, Slot
from src.domain.schedule import assign_places

# 위도 0.01° ≈ 1.1km
ANCHOR = (35.0, 129.0)


def P(name, cat, lat=None, lng=129.0):
    return Place(name=name, region="부산광역시 중구", category=cat, lat=lat, lng=lng if lat is not None else None)


def slot(sid, day, order, cat):
    return Slot(slot_id=sid, day=day, order=order, category=cat)


def test_without_coords_keeps_rank_order():
    slots = [slot("d1-s1", 1, 1, "식당"), slot("d1-s2", 1, 2, "식당")]
    picks, _ = assign_places(slots, {"식당": [P("A", "식당"), P("B", "식당")]})
    assert [picks[s].name for s in ("d1-s1", "d1-s2")] == ["A", "B"]


def test_prefers_near_candidate_over_slightly_better_ranked_far_one():
    slots = [slot("d1-s1", 1, 1, "식당")]
    cands = {"식당": [P("먼1위", "식당", 35.5), P("가까운2위", "식당", 35.01)]}   # 55km vs 1km
    picks, alts = assign_places(slots, cands, anchors={1: ANCHOR})
    assert picks["d1-s1"].name == "가까운2위"
    assert [p.name for p in alts["d1-s1"]] == ["먼1위"]      # 반경 밖은 대안에서도 뒤로


def test_chains_from_previous_place_not_only_anchor():
    slots = [slot("d1-s1", 1, 1, "식당"), slot("d1-s2", 1, 2, "카페")]
    cands = {"식당": [P("북쪽식당", "식당", 35.05)],
             "카페": [P("앵커옆카페", "카페", 35.0), P("식당옆카페", "카페", 35.051)]}
    picks, _ = assign_places(slots, cands, anchors={1: ANCHOR})
    assert picks["d1-s2"].name == "식당옆카페"


def test_relaxes_radius_when_nothing_is_near_and_never_duplicates():
    slots = [slot("d1-s1", 1, 1, "관광지"), slot("d2-s1", 2, 1, "관광지"), slot("d2-s2", 2, 2, "관광지")]
    cands = {"관광지": [P("먼곳", "관광지", 36.0), P("좌표없음", "관광지")]}
    picks, _ = assign_places(slots, cands, anchors={1: ANCHOR, 2: ANCHOR})
    names = [picks[s].name if picks[s] else None for s in ("d1-s1", "d2-s1", "d2-s2")]
    assert sorted(n for n in names if n) == ["먼곳", "좌표없음"] and names.count(None) == 1


def test_breakfast_slot_uses_breakfast_candidates():
    slots = [Slot(slot_id="d2-s1", day=2, order=1, category="식당", meal="아침"),
             Slot(slot_id="d2-s2", day=2, order=2, category="식당", meal="점심")]
    cands = {"식당": [P("국수집", "식당"), P("해장국집", "식당")], "아침": [P("해장국집", "식당")]}
    picks, _ = assign_places(slots, cands)
    assert picks["d2-s1"].name == "해장국집" and picks["d2-s2"].name == "국수집"


def test_refinement_removes_zigzag_by_looking_at_both_neighbors():
    # 부산역(앵커)에서 북쪽 서면이 조금 더 가깝지만, 이후 장소가 모두 남쪽(영도)이면 남쪽 식당이 맞다
    slots = [slot("d1-s1", 1, 1, "식당"), slot("d1-s2", 1, 2, "관광지"), slot("d1-s3", 1, 3, "카페")]
    cands = {"식당": [P("서면식당", "식당", 35.03), P("x", "식당", 36.5), P("남포식당", "식당", 34.965)],
             "관광지": [P("영도명소", "관광지", 34.95)], "카페": [P("영도카페", "카페", 34.945)]}
    picks, _ = assign_places(slots, cands, anchors={1: ANCHOR})
    assert picks["d1-s1"].name == "남포식당"
