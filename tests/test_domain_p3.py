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
    assert [(x.day, x.category, x.meal or x.when) for x in s] == [
        (1, "식당", "점심"), (1, "관광지", None), (1, "카페", None), (1, "식당", "저녁"), (1, "관광지", "밤"),
        (2, "식당", "아침"), (2, "관광지", None), (2, "식당", "점심"), (2, "관광지", None), (2, "카페", None),
        (2, "식당", "저녁"), (2, "관광지", "밤"),
        (3, "식당", "아침"), (3, "관광지", None), (3, "식당", "점심")]
    assert s[0].slot_id == "d1-s1"
    assert [x.category for x in build_slots(1)] == ["식당", "관광지", "카페", "식당", "관광지"]  # 당일치기 = 첫날
    assert [x.day for x in build_slots(2)] == [1] * 5 + [2] * 3                          # 1박 2일 = 첫날 + 마지막 날



def test_check_trip_times():
    from src.domain.schedule import check_trip_times
    assert check_trip_times(1, "12:00", "12:30") is not None          # 당일치기인데 30분
    assert check_trip_times(1, "14:00", "12:00") is not None          # 출발이 도착보다 이름
    assert check_trip_times(1, "12:00", "20:00") is None
    assert check_trip_times(2, "12:00", "09:30") is not None          # 마지막 날 아침도 못 먹고 떠남
    assert check_trip_times(2, "17:00", "11:00") is None


def test_night_sightseeing_slot_after_dinner_except_last_day():
    s = build_slots(3)
    night = [(x.day, x.order) for x in s if x.when == "밤"]
    assert night == [(1, 5), (2, 7)]
    assert [x.when for x in build_slots(1)][-1] == "밤"          # 당일치기도 (귀가 시각이 허락하면)
