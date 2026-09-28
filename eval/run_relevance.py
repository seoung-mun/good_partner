"""검색 결과 파일의 상위 k개를 취향 적합도 judge로 채점 (관련도 지표)."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from eval.metrics import mean_or_none


def relevance_summary(rows: list[dict]) -> dict:
    per_item = [mean_or_none(r["scores"]) for r in rows]
    flat = [s for r in rows for s in r["scores"] if s is not None]
    return {"n_items": len(rows), "n_scored_items": sum(v is not None for v in per_item),
            "mean_relevance": mean_or_none(per_item),
            "relevant_rate": (sum(s >= 4 for s in flat) / len(flat)) if flat else None}


async def main(args):
    load_dotenv(".env")
    from eval.instrument import configure_caches
    from eval.judge import build_judge_chain, judge_place
    from src.llm_factory import make_llm
    cache_state = configure_caches(args.no_cache)
    model = os.getenv("JUDGE_MODEL") or os.environ["LLM_MODEL"]
    chain = build_judge_chain(make_llm(model))
    res = json.loads(Path(args.results).read_text())
    rows = []
    for n, item in enumerate(res["rows"], 1):
        scores = []
        for c in item.get("top", [])[: args.k]:
            r = await judge_place(chain, item, {"name": c["name"], "type": c["category"], "address": c["region"]})
            scores.append(r["score"])
        rows.append({"id": item["id"], "scores": scores})
        print(f"[{n}/{len(res['rows'])}] {item['id']} {scores}", flush=True)
    out = {"judge_model": model, "k": args.k, "cache": cache_state, "source": args.results,
           "summary": relevance_summary(rows), "rows": rows}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(out["summary"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--no-cache", action="store_true")
    asyncio.run(main(ap.parse_args()))
