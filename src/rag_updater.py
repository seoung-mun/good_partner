"""사용자 리뷰 누적분을 벡터 인덱스에 반영.

원본 리뷰는 reviews_log.csv에 추가만 하고(증거), 인덱스는 새 폴더에 만들어 검증을 통과한 뒤에만 교체한다.
운영 인덱스를 직접 고치지 않으므로 실패해도 기존 인덱스는 그대로이고, 직전 버전은 <인덱스>.prev-<시각>에 남는다.
"""
import os
import re
import time

import emoji
import pandas as pd
from langchain_core.documents import Document

from src.config import review_faiss
from src.errors import AppError
from src.logging_config import get_logger
from src.retrieval.store import load_faiss_index, make_embeddings

log = get_logger(__name__)

MIN_BATCH = 10
REVIEW_LOG = "reviews_log.csv"


class IndexUpdateError(AppError):
    """리뷰를 인덱스에 반영하지 못함 (기존 인덱스는 바뀌지 않음)."""

    code = "INDEX_UPDATE_FAILED"
    user_message = "리뷰는 저장됐지만 검색 데이터에 반영하지 못했어요. 기존 데이터는 그대로예요."


def clean_review(text):
    """리뷰 텍스트 정제: 공백 정규화, 이모지·특수문자 제거."""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    return text.strip()


def chunk_text_with_overlap(text, chunk_size=500, overlap=50):
    """긴 텍스트를 chunk_size 글자 단위로 overlap만큼 겹쳐 분할."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
        if start >= len(text): break
    return chunks


def find_address_from_db(db, place_name):
    """기존 인덱스에서 장소명이 겹치는 문서의 상세 주소 (없으면 빈 문자열)."""
    results = db.similarity_search(place_name, k=1)
    if not results:
        return ""
    existing_name = results[0].metadata.get("장소명", "")
    if place_name in existing_name or existing_name in place_name:
        return results[0].metadata.get("상세 주소", "")
    return ""


def create_documents_from_df(df, existing_db=None):
    """
    DataFrame -> Document 변환
    * existing_db: 주소 조회를 위해 전달받은 기존 FAISS DB 객체
    """
    docs = []
    for _, row in df.iterrows():
        cleaned_review = clean_review(row.get("리뷰", ""))
        chunks = chunk_text_with_overlap(cleaned_review, chunk_size=500, overlap=20)

        place_name = row.get("장소명") if pd.notna(row.get("장소명")) else row.get("장소", "장소미상")
        category = row.get("카테고리_통합") if pd.notna(row.get("카테고리_통합")) else row.get("카테고리", "기타")
        rating = row.get("평점") if pd.notna(row.get("평점")) else row.get("별점", "0")

        address = row.get("상세 주소") if pd.notna(row.get("상세 주소")) else ""
        if not address and existing_db:
            address = find_address_from_db(existing_db, place_name)

        for chunk in chunks:
            if len(chunk) <= 5: continue

            combined_text = (
                f"지역: {row.get('지역', '')} | "
                f"장소명: {place_name} | "
                f"카테고리: {category} | "
                f"리뷰: {chunk}"
            )
            docs.append(Document(
                page_content=combined_text,
                metadata={
                    "지역": str(row.get("지역", "")),
                    "카테고리": str(category),
                    "장소명": str(place_name),
                    "별점": str(rating),
                    "상세 주소": str(address),
                    "리뷰": str(row.get("리뷰", "")[:100])
                }
            ))
    return docs


def _append_log(df: pd.DataFrame) -> None:
    """원본 리뷰를 추가 전용 로그에 남긴다 (인덱스를 다시 만들 때의 원본)."""
    df.to_csv(REVIEW_LOG, mode="a", header=not os.path.exists(REVIEW_LOG), index=False, encoding="utf-8-sig")


def _verify(path: str, embeddings, expected: int) -> None:
    """새 인덱스를 다시 읽어 벡터 수와 문서 수가 기대값과 맞는지 확인."""
    from langchain_community.vectorstores import FAISS
    db = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    n_vec, n_doc = db.index.ntotal, len(db.index_to_docstore_id)
    if n_vec != expected or n_doc != expected:
        raise IndexUpdateError(f"검증 실패: 벡터 {n_vec}, 문서 {n_doc}, 기대 {expected}")


def update_vector_db_if_needed(new_reviews_file: str = "new_reviews.csv", index_dir: str = review_faiss) -> str:
    """누적 리뷰가 MIN_BATCH개 이상이면 검증된 새 인덱스로 교체하고 누적 파일을 비운다.

    반영 실패는 IndexUpdateError 등 예외로 던진다(기존 인덱스·누적 파일은 그대로).
    """
    try:
        df = pd.read_csv(new_reviews_file)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return "업데이트할 리뷰가 없습니다."
    if len(df) < MIN_BATCH:
        return f"리뷰 {len(df)}개 누적됨. ({MIN_BATCH}개 이상이어야 업데이트)"
    if not os.path.isdir(index_dir):
        raise IndexUpdateError(f"기존 인덱스가 없습니다: {index_dir}")

    from langchain_community.vectorstores import FAISS
    log.info("리뷰 %d개 인덱스 반영 시작", len(df))
    embeddings = make_embeddings()
    db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    before = db.index.ntotal
    new_docs = create_documents_from_df(df, existing_db=db)
    if not new_docs:
        raise IndexUpdateError("정제 후 남은 리뷰 문서가 없습니다 (너무 짧은 리뷰뿐)")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    staging = f"{index_dir}.staging-{stamp}"
    db.add_documents(new_docs)
    db.save_local(staging)
    _verify(staging, embeddings, before + len(new_docs))
    _append_log(df)
    os.replace(index_dir, f"{index_dir}.prev-{stamp}")   # 직전 버전 보관 (지우지 않음)
    os.replace(staging, index_dir)
    os.remove(new_reviews_file)                           # 원본은 이미 REVIEW_LOG에 있음
    load_faiss_index.cache_clear()
    log.info("인덱스 교체 완료: %d → %d", before, before + len(new_docs))
    return f"리뷰 {len(df)}개를 검색 데이터에 반영했어요."
