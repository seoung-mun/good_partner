import pytest
from eval.build_golden import build_items

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def items():
    return build_items()


def test_groups_present(items):
    groups = {i["group"] for i in items}
    assert groups == {"ambiguous", "gwangju", "regular", "landmark"}


def test_gwangju_pair(items):
    g = [i for i in items if i["group"] == "gwangju"]
    assert {(i["destination"], i["gold_sido"], i["gold_sigungu"]) for i in g} == {
        ("광주", "광주광역시", None), ("경기 광주", "경기도", "광주시")}
    assert len(g) == 6


def test_ambiguous_covers_same_named_districts(items):
    amb = {(i["gold_sido"], i["gold_sigungu"]) for i in items if i["group"] == "ambiguous"}
    assert ("부산광역시", "중구") in amb and ("대구광역시", "중구") in amb
    assert ("강원특별자치도", "고성군") in amb and ("경상남도", "고성군") in amb


def test_no_seoul_short_duplicates(items):
    assert all(i["gold_sido"] != "서울" for i in items)


def test_ids_unique_and_deterministic(items):
    assert len({i["id"] for i in items}) == len(items)
    assert [i["id"] for i in build_items()] == [i["id"] for i in items]


def test_every_gold_has_enough_places(items):
    from eval.metadata import place_counts
    c = place_counts()
    for i in items:
        if i["gold_sigungu"] is None:
            continue
        assert c.get((i["gold_sido"], i["gold_sigungu"], i["gold_category"]), 0) >= 5, i["id"]
