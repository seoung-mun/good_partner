from langchain_core.documents import Document

from src.agent.tools.place import find_and_select_best_place
from src.retrieval.retriever import _extract_reviews


def test_exclude_places_default_is_not_shared_list():
    assert find_and_select_best_place.coroutine.__defaults__[1] is None


def test_extract_reviews_from_metadata():
    doc = Document(page_content="지역: 부산광역시 중구 | 장소명: A | 카테고리: 식당 | 리뷰: 맛있어요",
                   metadata={"리뷰": "맛있어요 정말로"})
    assert _extract_reviews(doc) == ["맛있어요 정말로"]


def test_extract_reviews_prefers_review_list():
    doc = Document(page_content="x", metadata={"reviews": ["a", "b", "c", "d"], "리뷰": "z"})
    assert _extract_reviews(doc) == ["a", "b", "c"]


def test_extract_reviews_empty():
    assert _extract_reviews(Document(page_content="x", metadata={})) == []
