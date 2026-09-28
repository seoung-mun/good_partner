from eval.regions import canonical_region, region_match, SIDO_SHORT


def test_seoul_short_and_full_are_same():
    assert canonical_region("서울 강동구") == ("서울특별시", "강동구")
    assert canonical_region("서울특별시 강동구") == ("서울특별시", "강동구")


def test_gwangju_is_not_gyeonggi_gwangju():
    assert canonical_region("광주") == ("광주광역시", None)
    assert canonical_region("경기 광주") == ("경기도", "광주시")
    assert not region_match("경기도 광주시", "광주광역시", None)
    assert not region_match("광주광역시 동구", "경기도", "광주시")
    assert region_match("광주광역시 동구", "광주광역시", None)


def test_same_sigungu_name_different_sido():
    assert not region_match("대구광역시 중구", "부산광역시", "중구")
    assert region_match("부산광역시 중구", "부산광역시", "중구")


def test_renamed_provinces():
    assert canonical_region("강원도 강릉시") == ("강원특별자치도", "강릉시")
    assert canonical_region("전라북도 전주시") == ("전북특별자치도", "전주시")


def test_sigungu_short_form_matches():
    # 사용자는 "경기 광주"처럼 '시'를 생략한다
    assert canonical_region("경기 광주") == ("경기도", "광주시")
    assert canonical_region("부산 중구") == ("부산광역시", "중구")


def test_unknown_returns_raw():
    assert canonical_region("") == ("", None)
    assert region_match("", "부산광역시", None) is False


def test_short_names_cover_all_sido():
    assert SIDO_SHORT["경기도"] == "경기"
    assert len(SIDO_SHORT) == 17
