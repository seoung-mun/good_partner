import os

EMBED_MODEL = "upskyy/bge-m3-korean"


def pick_device() -> str:
    """쿼리 임베딩은 CPU에서 4~11초, MPS에서 0.17초 (결과 벡터 동일). 가능한 가속기를 쓴다."""
    if os.getenv("EMBED_DEVICE"):
        return os.environ["EMBED_DEVICE"]
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    # 인덱스(scripts/reembed.py)와 동일하게 비정규화 — 정규화 여부가 다르면 L2 거리 기준이 어긋남
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": pick_device()},
                                 encode_kwargs={"normalize_embeddings": False})
