import argparse
import asyncio
import csv
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

from eval.instrument import enable_llm_cache
from eval.judge import build_judge_chain, judge_place
from eval.legacy_adapter import build_preference
from eval.metadata import reviews_for
from eval.metrics import mean_or_none


async def main(args):
    load_dotenv(".env")
    if not args.no_cache:
        enable_llm_cache()
    from src.llm_factory import make_llm
    model = os.getenv("JUDGE_MODEL") or os.environ["LLM_MODEL"]
    chain = build_judge_chain(make_llm(model))
    e2e = json.loads(Path(args.e2e).read_text())
    judged = []
    for trip in e2e["rows"]:
        for place in [x for x in trip["itinerary"] if x.get("type") != "move"]:
            r = await judge_place(chain, trip, place)
            judged.append({"trip_id": trip["id"], "name": place.get("name"),
                           "region": place.get("address", ""), "category": place.get("type", ""), **r})
            print(f"{trip['id']} {place.get('name')}: {r['status']} {r['score']}")

    status = {s: sum(j["status"] == s for j in judged) for s in ("ok", "unmatched", "parse_error")}
    per_trip = {}
    for j in judged:
        per_trip.setdefault(j["trip_id"], []).append(j["score"])
    out = {"judge_model": model, "status_counts": status,
           "mean_score": mean_or_none([j["score"] for j in judged]),
           "mean_trip_score": mean_or_none([mean_or_none(v) for v in per_trip.values()]),
           "rows": judged}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print({k: out[k] for k in ("judge_model", "status_counts", "mean_score", "mean_trip_score")})

    if args.sheet and not Path(args.sheet).exists():
        trips = {t["id"]: t for t in e2e["rows"]}
        scored = [j for j in judged if j["status"] == "ok"]
        sample = random.Random(0).sample(scored, min(40, len(scored)))
        with open(args.sheet, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["idx", "trip_id", "request", "place", "category", "region", "reviews", "human_score"])
            for n, j in enumerate(sample):
                w.writerow([n, j["trip_id"], build_preference(trips[j["trip_id"]]), j["name"], j["category"],
                            j["region"], "\n".join(reviews_for(j["name"], j["region"])), ""])
        print(f"수기 라벨 시트 → {args.sheet} (judge 점수는 숨김, human_score 열을 1~5로 채우세요)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="eval/results/e2e_baseline.json")
    ap.add_argument("--out", default="eval/results/judge_baseline.json")
    ap.add_argument("--sheet", default="eval/data/judge_labels.csv")
    ap.add_argument("--no-cache", action="store_true")
    asyncio.run(main(ap.parse_args()))
