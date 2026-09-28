"""metadata_list.pkl → FAISS 인덱스 재생성 (data_process.ipynb 셀 31~33 레시피 재현).

한 번의 forward로 두 인덱스를 만든다:
  review_faiss/       CLS 풀링 (원본 재현, 베이스라인)
  review_faiss_mean/  Mean 풀링 (앱 쿼리 임베딩과 같은 방식, 수정본)

중단돼도 다시 실행하면 data/emb_shards/ 에서 이어서 진행한다.
사용: .venv/bin/python scripts/reembed.py [--limit N]
"""
import argparse, os, pickle, time
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, "data", "metadata_list.pkl")
SHARDS = os.path.join(ROOT, "data", "emb_shards")
MODEL = "upskyy/bge-m3-korean"
MAX_LEN = 512       # 원본과 동일
BATCH = 32
SHARD_EVERY = 100   # 배치 100개(3,200건)마다 디스크에 저장


def build_store(meta, vecs, out_dir):
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    index = faiss.IndexFlatL2(vecs.shape[1])  # 원본과 동일, 비정규화
    index.add(vecs)
    docs = {str(i): Document(page_content=r["text_for_embedding"],
                             metadata={k: v for k, v in r.items() if k != "text_for_embedding"})
            for i, r in enumerate(meta)}
    store = FAISS(embedding_function=None, index=index, docstore=InMemoryDocstore(docs),
                  index_to_docstore_id={i: str(i) for i in range(len(meta))})
    store.save_local(out_dir)
    print(f"saved {out_dir} ({index.ntotal} vectors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="앞 N건만 (스모크 테스트용)")
    args = ap.parse_args()

    meta = pickle.load(open(META, "rb"))[: args.limit]
    texts = [r["text_for_embedding"] for r in meta]
    n = len(texts)
    suffix = f"_limit{args.limit}" if args.limit else ""
    shards = SHARDS + suffix  # 스모크 테스트 shard가 본 실행과 섞이지 않게
    os.makedirs(shards, exist_ok=True)

    # 길이순 정렬 → 패딩 낭비 감소. 결과는 원래 인덱스 위치에 되돌려 넣음.
    order = np.argsort([len(t) for t in texts])
    batches = [order[i:i + BATCH] for i in range(0, n, BATCH)]
    shard_groups = [batches[i:i + SHARD_EVERY] for i in range(0, len(batches), SHARD_EVERY)]

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    print(f"{n} texts, {len(shard_groups)} shards, device={device}")

    t0 = time.time()
    for s, group in enumerate(shard_groups):
        path = os.path.join(shards, f"{s:05d}.npz")
        if os.path.exists(path):
            continue
        ids, cls_out, mean_out = [], [], []
        with torch.no_grad():
            for b in group:
                enc = tok([texts[i] for i in b], return_tensors="pt", truncation=True,
                          padding=True, max_length=MAX_LEN).to(device)
                h = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                cls_out.append(h[:, 0].float().cpu().numpy())
                mean_out.append(((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
                ids.append(b)
        np.savez(path + ".tmp.npz", ids=np.concatenate(ids),
                 cls=np.concatenate(cls_out), mean=np.concatenate(mean_out))
        os.replace(path + ".tmp.npz", path)  # 원자적 저장: 중간에 죽어도 깨진 shard 안 남음
        done = min((s + 1) * SHARD_EVERY * BATCH, n)
        el = time.time() - t0
        print(f"shard {s + 1}/{len(shard_groups)}  {done}/{n}  {el / 60:.1f}min", flush=True)

    cls = np.zeros((n, 1024), dtype="float32")
    mean = np.zeros((n, 1024), dtype="float32")
    for f in sorted(os.listdir(shards)):
        z = np.load(os.path.join(shards, f))
        cls[z["ids"]] = z["cls"]
        mean[z["ids"]] = z["mean"]
    assert not (np.abs(cls).sum(1) == 0).any(), "빈 벡터 존재 — shard 누락"

    build_store(meta, cls, os.path.join(ROOT, "review_faiss" + suffix))
    build_store(meta, mean, os.path.join(ROOT, "review_faiss_mean" + suffix))


if __name__ == "__main__":
    main()
