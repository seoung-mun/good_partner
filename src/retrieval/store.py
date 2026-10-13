"""벡터 스토어: 임베딩 모델 생성과 FAISS 인덱스 로드."""
import os
from functools import lru_cache

from src.config import review_faiss
from src.logging_config import get_logger

log = get_logger(__name__)

EMBED_MODEL = "upskyy/bge-m3-korean"


def pick_device() -> str:
    """서비스 배포 대상은 CPU 서버(의도된 설계). 개발·평가 시에만 EMBED_DEVICE=mps|cuda로 가속기 사용."""
    return os.getenv("EMBED_DEVICE", "cpu")


def make_embeddings():
    """문서 인덱스와 같은 설정(bge-m3-korean, 비정규화)의 쿼리 임베딩 모델."""
    from langchain_huggingface import HuggingFaceEmbeddings
    # 인덱스(scripts/reembed.py)와 동일하게 비정규화 — 정규화 여부가 다르면 L2 거리 기준이 어긋남
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": pick_device()},
                                 encode_kwargs={"normalize_embeddings": False})


@lru_cache(maxsize=1)
def load_faiss_index():
    """FAISS 인덱스를 한 번만 로드해 프로세스 안에서 공유한다. 로드 실패는 예외로 그대로 올린다."""
    from langchain_community.vectorstores import FAISS
    log.info("임베딩 모델 및 FAISS 인덱스 로딩 중: %s", review_faiss)
    db = FAISS.load_local(review_faiss, make_embeddings(), allow_dangerous_deserialization=True)
    log.info("FAISS 인덱스 로딩 완료 (%d 벡터)", db.index.ntotal)
    return db
