import argparse
import csv
import json
from pathlib import Path

from eval.metrics import weighted_kappa


def main(args):
    judged = {(j["trip_id"], j["name"]): j["score"]
              for j in json.loads(Path(args.judge).read_text())["rows"] if j["score"] is not None}
    human, model = [], []
    with open(args.labels, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["human_score"].strip():
                key = (row["trip_id"], row["place"])
                if key in judged:
                    human.append(int(row["human_score"]))
                    model.append(judged[key])
    exact = sum(h == m for h, m in zip(human, model)) / len(human)
    within1 = sum(abs(h - m) <= 1 for h, m in zip(human, model)) / len(human)
    print(json.dumps({"n": len(human), "quadratic_weighted_kappa": round(weighted_kappa(human, model), 3),
                      "exact_agreement": round(exact, 3), "within_1": round(within1, 3)}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="eval/results/judge_baseline.json")
    ap.add_argument("--labels", default="eval/data/judge_labels.csv")
    main(ap.parse_args())
