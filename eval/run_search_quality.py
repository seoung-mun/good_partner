"""검색 품질 측정: 실버 골든셋(MRR@10·nDCG@10·Recall@20) + 기존 지역 골든셋(지역 정확도·빈 결과율).

LLM 없음 — src.retrieval.retriever.search_candidates를 직접 호출. 단계별 커밋에서 같은 명령으로 실행해 전후 비교.
"""
import argparse
import json
import time
from pathlib import Path

import eval  # noqa: F401  (OMP 설정)
from eval.metrics import aggregate, mrr_at_k, ndcg_at_k, recall_at_k, retrieval_item_metrics

RANK_KEYS = ["mrr@10", "ndcg@10", "recall@20", "latency_s"]
REGION_KEYS = ["empty", "returned", "region_acc", "region_hit1", "category_acc"]
BASE_WORD = {"식당": "맛집", "카페": "카페", "관광지": "관광지"}


def main(args):
    from src.domain.region import resolve_destination
    from src.retrieval.retriever import search_candidates
    from src.retrieval.store import load_faiss_index
    db = load_faiss_index()

    rank_rows = []
    for it in [json.loads(l) for l in Path(args.search).read_text().splitlines() if l.strip()]:
        t0 = time.time()
        places = search_candidates(db, it["region"], [it["category"]], [it["keyword"]])[it["category"]]
        ranked = [(p.name, p.region) for p in places]
        rel = {tuple(x) for x in it["relevant"]}
        rank_rows.append({"id": it["id"], "category": it["category"], "returned": len(ranked),
                          "mrr@10": mrr_at_k(ranked, rel, 10), "ndcg@10": ndcg_at_k(ranked, rel, 10),
                          "recall@20": recall_at_k(ranked, rel, 20), "latency_s": time.time() - t0})

    region_rows = []
    for it in [json.loads(l) for l in Path(args.region).read_text().splitlines() if l.strip()]:
        resolved = resolve_destination(it["destination"])
        region = " ".join(x for x in resolved if x) if resolved else it["destination"]
        extra = [w for w in it["query"].split() if w != BASE_WORD[it["category_filter"]]]  # 지명 쿼리('해운대 맛집')
        places = search_candidates(db, region, [it["category_filter"]], extra)[it["category_filter"]]
        cands = [{"name": p.name, "region": p.region, "category": p.category} for p in places]
        region_rows.append({"id": it["id"], "group": it["group"], **retrieval_item_metrics(cands, it, args.k)})

    out = {"label": args.label,
           "summary": {"rank": aggregate(rank_rows, RANK_KEYS, by="category"),
                       "region": aggregate(region_rows, REGION_KEYS, by="group")},
           "rank_rows": rank_rows, "region_rows": region_rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", default="eval/data/golden_search.jsonl")
    ap.add_argument("--region", default="eval/data/golden_retrieval.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
