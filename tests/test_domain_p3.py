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
    assert [(x.day, x.category) for x in s] == [
        (1, "식당"), (1, "카페"), (1, "관광지"), (1, "식당"),
        (2, "관광지"), (2, "식당"), (2, "카페"), (2, "관광지"), (2, "식당"),
        (3, "관광지"), (3, "식당"), (3, "카페")]
    assert s[0].slot_id == "d1-s1" and s[0].meal == "점심" and s[3].meal == "저녁"
    assert [x.category for x in build_slots(1)] == ["식당", "카페", "관광지", "식당"]

