import argparse
import asyncio
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from eval.instrument import LLMCallCounter, attach_counter, configure_caches
from eval.metrics import e2e_trip_metrics, mean_or_none, summarize_e2e
from eval.trace import AgentTracer, timed

KEYS = ["n_places", "slot_fill", "category_adherence", "overflow", "region_acc", "dup_rate",
        "has_timeline", "total_move_min", "llm_calls", "input_tokens", "output_tokens", "maps_calls", "latency_s",
        "ungrounded_rate"]
# 실패 요청도 서비스 품질에 포함되는 지표 (전체 평균)
SERVICE_KEYS = ["tool_error_rate", "invalid_tool_call_rate", "redundant_tool_calls", "tool_soft_errors",
                "llm_time_s", "search_time_s", "maps_time_s", "other_time_s"]


def resummarize(path: str) -> None:
    """저장된 결과로 행별 지표와 summary를 다시 계산 (LLM 재실행 없음)."""
    d = json.loads(Path(path).read_text())
    for r in d["rows"]:  # 지표 정의가 바뀌어도 저장된 itinerary로 행별 지표를 다시 계산
        r.update(e2e_trip_metrics(r["itinerary"], r))
    d["summary"] = {"all": summarize_e2e(d["rows"], KEYS, SERVICE_KEYS)}
    Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(d["summary"], ensure_ascii=False, indent=1))


async def main(args):
    load_dotenv(".env")
    cache_state = configure_caches(args.no_cache)
    from src.config import LLM
    from src.graph_flow import build_graph
    from src.maps_compat import RoutesCompatClient
    from eval.legacy_adapter import run_legacy_trip
    from eval.metadata import reviews_for
    import src.tools as T
    from src.tools import AVAILABLE_TOOLS
    search_acc = {"t": 0.0, "n": 0}
    T._search_docs = timed(T._search_docs, search_acc)  # 임베딩+FAISS 시간 계측 (로직 동일)

    counter = LLMCallCounter()
    attach_counter(LLM, counter)
    app = build_graph()
    start = date.today() + timedelta(days=1)
    trips = [json.loads(l) for l in Path(args.golden).read_text().splitlines() if l.strip()][: args.limit]
    rows = []
    for n, trip in enumerate(trips, 1):
        counter.reset()
        m0, net0, mt0 = RoutesCompatClient.calls, RoutesCompatClient.network_calls, RoutesCompatClient.network_seconds
        search_acc.update(t=0.0, n=0)
        tracer = AgentTracer(known_tools=set(AVAILABLE_TOOLS))
        t0 = time.time()
        r = await run_legacy_trip(app, trip, start, callbacks=[tracer])
        latency = time.time() - t0
        tr = tracer.summary()
        places = [x for x in r["itinerary"] if x.get("type") != "move"]
        row = {**trip, **r, "latency_s": latency, **counter.snapshot(), **tr,
               "maps_calls": RoutesCompatClient.calls - m0,
               "maps_network_calls": RoutesCompatClient.network_calls - net0,
               "maps_time_s": RoutesCompatClient.network_seconds - mt0,
               "search_time_s": search_acc["t"], "search_calls": search_acc["n"],
               # DB에 없는 장소 = 검색 결과가 아닌 LLM이 지어낸 장소
               "ungrounded_rate": mean_or_none([float(not reviews_for(p.get("name", ""), p.get("address", "")))
                                                for p in places]),
               **e2e_trip_metrics(r["itinerary"], trip)}
        row["other_time_s"] = latency - tr["llm_time_s"] - row["search_time_s"] - row["maps_time_s"]
        row["ok"] = float(row["ok"])
        rows.append(row)
        print(f"[{n}/{len(trips)}] {trip['id']} ok={r['ok']} places={row['n_places']} "
              f"region={row['region_acc']} llm={row['llm_calls']} {latency:.0f}s "
              f"(llm {tr['llm_time_s']:.0f}s, search {row['search_time_s']:.0f}s, maps {row['maps_time_s']:.0f}s) "
              f"tool_err={tr['tool_error_rate']}", flush=True)

    out = {"llm_model": os.environ["LLM_MODEL"], "start_date": start.isoformat(),
           "cache": cache_state, "summary": {"all": summarize_e2e(rows, KEYS, SERVICE_KEYS)}, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/data/golden_trips.jsonl")
    ap.add_argument("--out", default="eval/results/e2e_baseline.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--resummarize", metavar="RESULT_JSON")
    a = ap.parse_args()
    resummarize(a.resummarize) if a.resummarize else asyncio.run(main(a))
