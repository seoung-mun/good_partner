import pickle
import pytest

pytestmark = pytest.mark.slow


def test_query_embedding_matches_index_space():
    """앱의 쿼리 임베딩(Mean 풀링)으로 문서 원문을 검색하면 그 문서 자신이 1위여야 한다 (문서·쿼리 풀링 일치)."""
    from src.retrieval.store import load_faiss_index
    meta = pickle.load(open("data/metadata_list.pkl", "rb"))
    db = load_faiss_index()
    for i in (5, 123456):
        doc, dist = db.similarity_search_with_score(meta[i]["text_for_embedding"], k=1)[0]
        assert doc.page_content == meta[i]["text_for_embedding"]
        assert dist < 1e-3
