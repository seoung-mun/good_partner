import pytest
from eval.metadata import place_counts, reviews_for

pytestmark = pytest.mark.slow


def test_place_counts_canonicalized():
    c = place_counts()
    assert c[("서울특별시", "강동구", "식당")] > 0
    assert ("서울", "강동구", "식당") not in c


def test_reviews_for_known_place():
    revs = reviews_for("이정태본참치", "부산광역시 중구")
    assert revs and all(isinstance(r, str) for r in revs)
    assert reviews_for("없는장소이름", "부산광역시 중구") == []


def test_ungrounded_rate():
    from eval.metadata import ungrounded_rate
    places = [{"name": "이정태본참치", "address": "부산광역시 중구"}, {"name": "지어낸가게", "address": "부산광역시 중구"}]
    assert ungrounded_rate(places) == 0.5
    assert ungrounded_rate([]) is None
