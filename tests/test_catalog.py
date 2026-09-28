from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.retrieval.catalog import bayes_rating, get_catalog


def md(name, region, cat, review, rating=4.0):
    return {"장소명": name, "지역": region, "카테고리": cat, "상세 주소": "주소", "별점": rating, "리뷰": review}


def make_db(rows):
    return FAISS.from_texts([r["리뷰"] for r in rows], DeterministicFakeEmbedding(size=8), metadatas=rows)


def test_region_ids_separate_same_name_regions():
    db = make_db([md("A", "광주광역시 동구", "식당", "r1"), md("B", "경기도 광주시", "식당", "r2"),
                  md("C", "광주광역시 서구", "카페", "r3")])
    cat = get_catalog(db)
    names = lambda ids: sorted(cat.meta[i]["장소명"] for i in ids)  # noqa: E731
    assert names(cat.ids_for_region("광주광역시")) == ["A", "C"]
    assert names(cat.ids_for_region("경기도 광주시")) == ["B"]
    assert names(cat.ids_for_region("광주광역시 동구")) == ["A"]


def test_place_stats_group_reviews_and_cache():
    db = make_db([md("A", "부산광역시 중구", "식당", "맛있다", 4.6), md("A", "부산광역시 중구", "식당", "친절"),
                  md("B", "부산광역시 중구", "식당", "보통", 3.0)])
    cat = get_catalog(db)
    a = cat.places[("A", "부산광역시 중구")]
    assert a.n_reviews == 2 and sorted(a.reviews) == ["맛있다", "친절"] and a.rating == 4.6
    assert get_catalog(db) is cat                       # 인덱스당 1회만 구축


def test_bayes_rating_shrinks_few_reviews_toward_mean():
    assert bayes_rating(5.0, 1, C=4.0, m=10) < bayes_rating(5.0, 100, C=4.0, m=10) < 5.0
    assert bayes_rating(None, 3, C=4.0, m=10) == 4.0


def test_nan_reviews_are_skipped():
    db = make_db([md("A", "부산광역시 중구", "식당", "좋아요")])
    db.docstore.search(db.index_to_docstore_id[0]).metadata["리뷰"] = float("nan")
    assert get_catalog(db).places[("A", "부산광역시 중구")].reviews == []
