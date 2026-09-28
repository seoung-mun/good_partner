import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from eval.instrument import LLMCallCounter, attach_counter, configure_caches
from eval.metrics import aggregate, retrieval_item_metrics

KEYS = ["empty", "returned", "region_acc", "region_hit1", "category_acc", "diversity", "llm_calls", "latency_s"]


async def main(args):
    load_dotenv(".env")
    cache_state = configure_caches(args.no_cache)
    from src.config import LLM
    from eval.legacy_adapter import doc_to_cand, legacy_retrieve
    counter = LLMCallCounter()
    attach_counter(LLM, counter)

    items = [json.loads(l) for l in Path(args.golden).read_text().splitlines() if l.strip()][: args.limit]
    rows = []
    for n, it in enumerate(items, 1):
        counter.reset()
        t0 = time.time()
        try:
            r = await legacy_retrieve(it["query"], it["destination"], it["category_filter"],
                                      user_info=it.get("user_info", ""))
            cands = [doc_to_cand(d) for d in r["candidates"]]
            row = {**it, "ok": True, "target_region": r["target_region"], "queries": r["queries"],
                   "fallback_used": r["fallback_used"], "top": cands[: args.k],
                   **retrieval_item_metrics(cands, it, args.k)}
        except Exception as e:
            row = {**it, "ok": False, "error": repr(e), **retrieval_item_metrics([], it, args.k)}
        row.update(latency_s=time.time() - t0, **counter.snapshot())
        rows.append(row)
        print(f"[{n}/{len(items)}] {it['id']}  region_acc={row['region_acc']:.2f}  cat={row['category_acc']:.2f}")

    out = {"llm_model": os.environ["LLM_MODEL"], "k": args.k, "cache": cache_state,
           "summary": aggregate(rows, KEYS, by="group"), "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/data/golden_retrieval.jsonl")
    ap.add_argument("--out", default="eval/results/retrieval_baseline.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    asyncio.run(main(ap.parse_args()))
