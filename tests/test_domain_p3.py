import pytest

from src.domain.region import resolve_destination
from src.domain.schedule import build_slots


@pytest.mark.parametrize("text,expected", [
    ("광주", ("광주광역시", "")),
    ("경기 광주", ("경기도", "광주시")),
    ("부산 중구", ("부산광역시", "중구")),
    ("울산 동구", ("울산광역시", "동구")),       # LLM 정규화는 시군구를 잃었던 케이스
    ("강릉", ("강원특별자치도", "강릉시")),      # 시도 없이 시군구만
    ("가평", ("경기도", "가평군")),
    ("제주", ("제주특별자치도", "")),
    ("서울 종로구", ("서울특별시", "종로구")),
    ("중구", None),                              # 여러 시도에 있음 → 결정 불가
    ("어딘가", None),
])
def test_resolve_destination(text, expected):
    assert resolve_destination(text) == expected


def test_build_slots_template():
    s = build_slots(3)
    assert [(x.day, x.category, x.meal) for x in s] == [
        (1, "식당", "점심"), (1, "관광지", None), (1, "카페", None), (1, "식당", "저녁"),
        (2, "식당", "아침"), (2, "관광지", None), (2, "식당", "점심"), (2, "관광지", None), (2, "카페", None),
        (2, "식당", "저녁"),
        (3, "식당", "아침"), (3, "관광지", None), (3, "식당", "점심")]
    assert s[0].slot_id == "d1-s1"
    assert [x.category for x in build_slots(1)] == ["식당", "관광지", "카페", "식당"]     # 당일치기 = 첫날
    assert [x.day for x in build_slots(2)] == [1] * 4 + [2] * 3                          # 1박 2일 = 첫날 + 마지막 날

