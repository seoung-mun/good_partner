import pytest
from langchain_core.runnables import RunnableLambda

from eval import judge
from eval.judge import JudgeScore, judge_place

TRIP = {"destination": "부산", "start_location": "", "days": 1, "group_type": "친구",
        "style": "맛집 탐방", "detail": "노포"}


@pytest.fixture
def reviews(monkeypatch):
    monkeypatch.setattr(judge, "reviews_for", lambda name, region, limit=5: ["오래된 노포, 맛있음"] if name == "A" else [])


async def test_ok(reviews):
    chain = RunnableLambda(lambda x: JudgeScore(score=4, rationale="노포 언급"))
    r = await judge_place(chain, TRIP, {"name": "A", "type": "식당", "address": "부산광역시 중구"})
    assert r == {"score": 4, "rationale": "노포 언급", "status": "ok"}


async def test_unmatched_place_is_not_scored(reviews):
    chain = RunnableLambda(lambda x: pytest.fail("호출되면 안 됨"))
    for name in ["추천 장소 없음", "없는가게"]:
        r = await judge_place(chain, TRIP, {"name": name, "type": "식당", "address": "부산광역시 중구"})
        assert r["score"] is None and r["status"] == "unmatched"


async def test_parse_error_recorded(reviews):
    def boom(_):
        raise ValueError("bad json")
    r = await judge_place(RunnableLambda(boom), TRIP, {"name": "A", "type": "식당", "address": "부산광역시 중구"})
    assert r["score"] is None and r["status"] == "parse_error"
