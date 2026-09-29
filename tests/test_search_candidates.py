from src.domain.models import Place, Slot
from src.domain.schedule import assign_places
from src.retrieval.retriever import search_candidates
from tests.fakes import make_db, meta

# 쿼리·문서 벡터를 직접 지정: 쿼리와 가까울수록 상위
NEAR, MID, FAR = [0.0, 0, 0, 0], [1.0, 0, 0, 0], [5.0, 0, 0, 0]


def test_region_prefilter_excludes_same_named_other_region_and_other_category():
    rows = [meta("광주집", "광주광역시 동구", "식당", "r1"), meta("경기광주집", "경기도 광주시", "식당", "r2"),
            meta("광주카페", "광주광역시 동구", "카페", "r3")]
    db = make_db(rows, {"r1": FAR, "r2": NEAR, "r3": NEAR})
    out = search_candidates(db, "광주광역시", ["식당"], [], k=1)
    assert [p.name for p in out["식당"]] == ["광주집"]      # 전국에서 더 가까운 경기 광주·카페가 k를 먹지 않음


def test_place_score_uses_several_reviews_not_one_lucky_chunk():
    rows = [meta("한줄", "부산광역시 중구", "식당", "lucky"),
            meta("꾸준", "부산광역시 중구", "식당", "a"), meta("꾸준", "부산광역시 중구", "식당", "b"),
            meta("꾸준", "부산광역시 중구", "식당", "c"), meta("먼곳", "부산광역시 중구", "식당", "far")]
    db = make_db(rows, {"lucky": [0.0, 0, 0, 0], "a": [0.3, 0, 0, 0], "b": [0.3, 0, 0, 0], "c": [0.3, 0, 0, 0],
                        "far": FAR})   # 모자란 청크는 검색된 것 중 가장 먼 거리로 채움 → 한 줄 운빨 장소가 불리
    out = search_candidates(db, "부산광역시 중구", ["식당"], [])
    assert [p.name for p in out["식당"]][0] == "꾸준"
    assert sorted(out["식당"][0].reviews) == ["a", "b", "c"]  # 가까운 리뷰를 근거로


def test_single_batched_embedding_for_all_queries():
    rows = [meta("A", "부산광역시 중구", "식당", "ra"), meta("B", "부산광역시 중구", "카페", "rb")]
    db = make_db(rows)
    search_candidates(db, "부산광역시 중구", ["식당", "카페"], ["노포"])
    assert len(db.embedding_function.calls) == 1


def test_empty_region_returns_empty_lists():
    db = make_db([meta("A", "부산광역시 중구", "식당", "ra")])
    assert search_candidates(db, "제주특별자치도", ["식당"], []) == {"식당": []}


def test_assign_places_no_duplicates_across_slots():
    slots = [Slot(slot_id="d1-s1", day=1, order=1, category="식당"),
             Slot(slot_id="d1-s2", day=1, order=2, category="식당"),
             Slot(slot_id="d1-s3", day=1, order=3, category="식당")]
    p = lambda n: Place(name=n, region="r", category="식당")  # noqa: E731
    out, _ = assign_places(slots, {"식당": [p("A"), p("B")]})
    assert out["d1-s1"].name == "A" and out["d1-s2"].name == "B" and out["d1-s3"] is None


def test_preference_query_results_rank_before_base_only_results():
    rows = [meta("기본만", "부산광역시 중구", "식당", "base"), meta("취향", "부산광역시 중구", "식당", "pref")]
    db = make_db(rows, {"base": [0.0, 0, 0, 0], "pref": [1.0, 0, 0, 0],
                        "노포 좋은 맛집": [1.0, 0, 0, 0], "좋은 맛집": [0.0, 0, 0, 0]})
    out = search_candidates(db, "부산광역시 중구", ["식당"], ["노포"])["식당"]
    assert [p.name for p in out] == ["취향", "기본만"] and out[0].score > out[1].score


def test_place_carries_review_count_and_keyword_quote():
    rows = [meta("무명", "부산광역시 중구", "식당", "그냥", rating=3.0),
            meta("유명", "부산광역시 중구", "식당", "노포 분위기", rating=4.8)]
    rows += [meta("유명", "부산광역시 중구", "식당", f"리뷰{i}", rating=4.8) for i in range(20)]
    db = make_db(rows)
    out = {p.name: p for p in search_candidates(db, "부산광역시 중구", ["식당"], ["노포"])["식당"]}
    assert out["유명"].n_reviews == 21 and out["유명"].score > 0
    assert out["유명"].quote == "노포 분위기"               # 키워드가 들어간 리뷰를 인용
    assert out["무명"].quote == "그냥"                      # 없으면 가장 가까운 리뷰


def test_per_category_keywords_only_go_to_their_category_query():
    db = make_db([meta("A", "부산광역시 중구", "식당", "ra"), meta("B", "부산광역시 중구", "카페", "rb")])
    search_candidates(db, "부산광역시 중구", ["식당", "카페"], {"식당": ["흑돼지"], "카페": ["바다뷰"]})
    texts = db.embedding_function.calls[0]
    assert "흑돼지 좋은 맛집" in texts and "바다뷰 좋은 카페" in texts
    assert not any("흑돼지" in t and "카페" in t for t in texts)
