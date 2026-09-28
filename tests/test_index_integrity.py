import pickle
import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def meta():
    return pickle.load(open("data/metadata_list.pkl", "rb"))


@pytest.fixture(scope="module")
def store():
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="upskyy/bge-m3-korean",
                                encode_kwargs={"normalize_embeddings": False})
    return FAISS.load_local("review_faiss", emb, allow_dangerous_deserialization=True)


def test_counts_match(meta, store):
    assert store.index.ntotal == len(meta) == 195787


def test_docstore_rows_match_metadata(meta, store):
    for i in (0, 1, 97000, 195786):
        doc = store.docstore.search(store.index_to_docstore_id[i])
        assert doc.page_content == meta[i]["text_for_embedding"]
        assert doc.metadata["지역"] == meta[i]["지역"]


def test_vectors_are_cls_of_row_text(meta, store):
    """저장 벡터 i == meta[i] 텍스트의 CLS 임베딩 (원본 레시피 재현 확인)."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("upskyy/bge-m3-korean")
    model = AutoModel.from_pretrained("upskyy/bge-m3-korean").eval()
    ids = [5, 123456]
    with torch.no_grad():
        enc = tok([meta[i]["text_for_embedding"] for i in ids], return_tensors="pt",
                  truncation=True, padding=True, max_length=512)
        cls = model(**enc).last_hidden_state[:, 0].numpy()
    for row, i in enumerate(ids):
        stored = store.index.reconstruct(i)
        cos = float(np.dot(stored, cls[row]) / np.linalg.norm(stored) / np.linalg.norm(cls[row]))
        assert cos > 0.999
