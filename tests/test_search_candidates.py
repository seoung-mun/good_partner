from langchain_core.documents import Document

from src.domain.models import Place, Slot
from src.domain.schedule import assign_places
from src.retrieval.retriever import search_candidates


def d(name, region, cat, review="좋아요"):
    return Document(page_content="x", metadata={"장소명": name, "지역": region, "카테고리": cat,
                                                "상세 주소": "주소", "별점": 4.5, "리뷰": review})


class FakeEmb:
    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(i)] for i in range(len(texts))]


class FakeDB:
    def __init__(self, by_text):
        self.embedding_function = FakeEmb()
        self.by_text = by_text
        self.texts = []

    def similarity_search_by_vector(self, vec, k=4):
        return self.by_text[self.texts[int(vec[0])]][:k]


def make_db(by_text):
    db = FakeDB(by_text)
    orig = db.embedding_function.embed_documents

    def embed(texts):
        db.texts = list(texts)
        return orig(texts)
    db.embedding_function.embed_documents = embed
    return db


def test_one_batched_embedding_and_filters():
    db = make_db({
        "부산광역시 중구 노포 맛집": [d("노포집", "부산광역시 중구", "식당"), d("타지집", "대구광역시 중구", "식당")],
        "부산광역시 중구 맛집": [d("일반집", "부산광역시 중구", "식당"), d("노포집", "부산광역시 중구", "식당", "두번째 리뷰"),
                             d("카페A", "부산광역시 중구", "카페")],
        "부산광역시 중구 노포 카페": [d("카페A", "부산광역시 중구", "카페")],
        "부산광역시 중구 카페": [],
    })
    out = search_candidates(db, "부산광역시 중구", ["식당", "카페"], ["노포"], k=10)
    assert len(db.embedding_function.calls) == 1                          # 배치 1회
    assert [p.name for p in out["식당"]] == ["노포집", "일반집"]            # 취향 쿼리 우선, 타 지역 제외, 중복 제거
    assert out["식당"][0].reviews == ["좋아요", "두번째 리뷰"]              # 같은 장소 리뷰는 모음
    assert [p.name for p in out["카페"]] == ["카페A"]                      # 카테고리 필터


def test_no_keywords_uses_base_query_only():
    db = make_db({"광주광역시 관광지": [d("공원", "광주광역시 동구", "관광지")]})
    out = search_candidates(db, "광주광역시", ["관광지"], [], k=10)
    assert db.embedding_function.calls == [["광주광역시 관광지"]]
    assert [p.name for p in out["관광지"]] == ["공원"]


def test_assign_places_no_duplicates_across_slots():
    slots = [Slot(slot_id="d1-s1", day=1, order=1, category="식당"),
             Slot(slot_id="d1-s2", day=1, order=2, category="식당"),
             Slot(slot_id="d1-s3", day=1, order=3, category="식당")]
    p = lambda n: Place(name=n, region="r", category="식당")
    out = assign_places(slots, {"식당": [p("A"), p("B")]})
    assert out["d1-s1"].name == "A" and out["d1-s2"].name == "B" and out["d1-s3"] is None
