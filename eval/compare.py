"""두 검색 결과 파일의 항목별 top-k 동일성과 지표 차이를 비교."""
import argparse
import json
from pathlib import Path


def diff_rows(a: dict, b: dict) -> list[str]:
    rb = {r["id"]: r for r in b["rows"]}
    return [r["id"] for r in a["rows"] if r.get("top") != rb.get(r["id"], {}).get("top")]


def main(args):
    a, b = (json.loads(Path(p).read_text()) for p in (args.before, args.after))
    changed = diff_rows(a, b)
    print(f"top-k가 달라진 항목: {len(changed)}/{len(a['rows'])}")
    for g in a["summary"]:
        sa, sb = a["summary"][g], b["summary"].get(g, {})
        print(f"{g:10s} " + "  ".join(f"{k} {sa[k]:.2f}→{sb.get(k, float('nan')):.2f}"
                                      for k in ("region_acc", "region_hit1", "category_acc", "empty")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    main(ap.parse_args())
