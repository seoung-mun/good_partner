"""테스트용 진짜 소형 FAISS 인덱스 (임베딩만 가짜). 검색 코드가 db.index·docstore를 직접 쓰므로 mock 대신 실물."""
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


class KeyedEmbedding(Embeddings):
    """텍스트 → 지정 벡터. 지정 없는 텍스트(쿼리 등)는 원점 — 쿼리 문구가 바뀌어도 테스트가 유지된다.
    embed_documents 호출을 기록."""

    def __init__(self, table: dict[str, list[float]] | None = None, dim: int = 4):
        self.table, self.dim, self.calls = table or {}, dim, []

    def _vec(self, t: str) -> list[float]:
        return self.table.get(t, [0.0] * self.dim)

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def meta(name, region, cat, review="좋아요", rating=4.0, address=None):
    """메타데이터 한 행 (크롤링 스키마와 같은 키)."""
    return {"장소명": name, "지역": region, "카테고리": cat, "상세 주소": address or f"{name} 주소",
            "별점": rating, "리뷰": review}


def make_db(rows: list[dict], table: dict[str, list[float]] | None = None):
    """rows의 '리뷰'를 문서 텍스트로 쓰는 FAISS. table로 문서·쿼리 벡터를 지정."""
    emb = KeyedEmbedding(table)
    db = FAISS.from_texts([r["리뷰"] for r in rows], emb, metadatas=rows)
    emb.calls.clear()
    return db
