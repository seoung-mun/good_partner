from src.domain.models import Place
from src.retrieval.timefit import TIME_PROTOTYPES, attach_time_fit
from tests.fakes import make_db, meta

NIGHT, DAY = [1.0, 0, 0, 0], [0, 1.0, 0, 0]


def test_time_fit_is_share_of_reviews_leaning_to_each_time():
    table = {s: NIGHT for s in TIME_PROTOTYPES["밤"]} | {s: DAY for s in TIME_PROTOTYPES["낮"]}
    table |= {"야경1": [1, 0.1, 0, 0], "야경2": [1, 0.2, 0, 0], "산책": [0.1, 1, 0, 0], "산책2": [0.2, 1, 0, 0]}
    db = make_db([meta("밤명소", "부산광역시 중구", "관광지", "야경1"), meta("밤명소", "부산광역시 중구", "관광지", "야경2"),
                  meta("밤명소", "부산광역시 중구", "관광지", "산책"), meta("공원", "부산광역시 중구", "관광지", "산책2")], table)
    out = attach_time_fit(db, [Place(name="밤명소", region="부산광역시 중구", category="관광지"),
                               Place(name="공원", region="부산광역시 중구", category="관광지")])
    assert abs(out[0].time_fit["밤"] - 2 / 3) < 1e-6 and out[1].time_fit["밤"] == 0.0


def test_time_fit_survives_zero_vectors_and_unknown_places():
    db = make_db([meta("A", "r", "관광지", "x")])                     # 모든 벡터가 원점
    out = attach_time_fit(db, [Place(name="A", region="r", category="관광지"),
                               Place(name="없음", region="r", category="관광지")])
    assert out[0].time_fit.get("밤", 0.0) == 0.0 and out[1].time_fit == {}
