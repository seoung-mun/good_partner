from eval.build_golden import relevant_places


def r(name, region, cat, review):
    return {"장소명": name, "지역": region, "카테고리": cat, "리뷰": review}


def test_relevant_places_needs_min_hits_in_region_and_category():
    rows = [r("A", "부산광역시 중구", "식당", "해산물 신선"), r("A", "부산광역시 중구", "식당", "해물탕 최고"),
            r("B", "부산광역시 중구", "식당", "해산물 좋음"),                       # 1회뿐
            r("C", "부산광역시 중구", "카페", "해산물"), r("C", "부산광역시 중구", "카페", "해물"),  # 카테고리 다름
            r("D", "대구광역시 중구", "식당", "해산물"), r("D", "대구광역시 중구", "식당", "해물")]  # 지역 다름
    assert relevant_places(rows, "부산광역시", "식당", ["해산물", "해물"], min_hits=2) == {("A", "부산광역시 중구")}
