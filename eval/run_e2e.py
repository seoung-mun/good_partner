import argparse
import asyncio
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from eval.instrument import LLMCallCounter, attach_counter, configure_caches
from eval.metrics import e2e_trip_metrics, summarize_e2e

KEYS = ["n_places", "slot_fill", "category_adherence", "overflow", "region_acc", "dup_rate",
        "total_move_min", "llm_calls", "input_tokens", "output_tokens", "maps_calls", "latency_s"]


async def main(args):
    load_dotenv(".env")
    cache_state = configure_caches(args.no_cache)
    from src.config import LLM
    from src.graph_flow import build_graph
    from src.maps_compat import RoutesCompatClient
    from eval.legacy_adapter import run_legacy_trip

    counter = LLMCallCounter()
    attach_counter(LLM, counter)
    app = build_graph()
    start = date.today() + timedelta(days=1)
    trips = [json.loads(l) for l in Path(args.golden).read_text().splitlines() if l.strip()][: args.limit]
    rows = []
    for n, trip in enumerate(trips, 1):
        counter.reset()
        m0, net0 = RoutesCompatClient.calls, RoutesCompatClient.network_calls
        t0 = time.time()
        r = await run_legacy_trip(app, trip, start)
        row = {**trip, **r, "latency_s": time.time() - t0, **counter.snapshot(),
               "maps_calls": RoutesCompatClient.calls - m0,
               "maps_network_calls": RoutesCompatClient.network_calls - net0,
               **e2e_trip_metrics(r["itinerary"], trip)}
        row["ok"] = float(row["ok"])
        rows.append(row)
        print(f"[{n}/{len(trips)}] {trip['id']} ok={r['ok']} places={row['n_places']} "
              f"region={row['region_acc']} llm={row['llm_calls']} {row['latency_s']:.0f}s")

    out = {"llm_model": os.environ["LLM_MODEL"], "start_date": start.isoformat(),
           "cache": cache_state, "summary": {"all": summarize_e2e(rows, KEYS)}, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/data/golden_trips.jsonl")
    ap.add_argument("--out", default="eval/results/e2e_baseline.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    asyncio.run(main(ap.parse_args()))
