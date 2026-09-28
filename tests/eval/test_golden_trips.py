import json
from pathlib import Path
from eval.regions import SIDO_SHORT

STYLES = {"맛집 탐방", "힐링/휴양", "액티비티", "문화/역사", "자연 감상"}


def load():
    return [json.loads(l) for l in Path("eval/data/golden_trips.jsonl").read_text().splitlines() if l.strip()]


def test_schema_and_values():
    trips = load()
    assert len(trips) == 20
    assert len({t["id"] for t in trips}) == 20
    for t in trips:
        assert 1 <= t["days"] <= 3
        assert t["style"] in STYLES
        assert t["gold_sido"] in SIDO_SHORT
        assert t["detail"]


def test_ambiguous_destinations_included():
    dests = {t["destination"] for t in load()}
    assert {"광주", "경기 광주"} <= dests
