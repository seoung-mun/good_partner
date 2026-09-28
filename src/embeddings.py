import os

EMBED_MODEL = "upskyy/bge-m3-korean"


def pick_device() -> str:
    """서비스 배포 대상은 CPU 서버(의도된 설계). 개발·평가 시에만 EMBED_DEVICE=mps|cuda로 가속기 사용."""
    return os.getenv("EMBED_DEVICE", "cpu")


def make_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    # 인덱스(scripts/reembed.py)와 동일하게 비정규화 — 정규화 여부가 다르면 L2 거리 기준이 어긋남
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": pick_device()},
                                 encode_kwargs={"normalize_embeddings": False})
