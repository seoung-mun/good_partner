"""성능 최적화 전후 비교: 지연·첫 일정 표시·단계별 시간 + 품질 유지 확인 → docs/eval/perf.md"""
import argparse
import json
from pathlib import Path

LATENCY = [("응답 지연 p50(초)", "latency_p50_all", 1), ("응답 지연 p95(초)", "latency_p95_all", 1),
           ("첫 일정 표시 p50(초)", "first_plan_p50", 1), ("첫 일정 표시 p95(초)", "first_plan_p95", 1),
           ("LLM 시간(초, 요청 평균)", "llm_time_s_all", 1), ("벡터 검색 시간(초)", "search_time_s_all", 1),
           ("Maps 시간(초)", "maps_time_s_all", 1), ("출력 토큰(성공 건 평균)", "output_tokens", 0)]
QUALITY = [("일정 생성 성공률", "success_rate", 2), ("일정 지역 정확도", "region_acc", 2), ("슬롯 충족률", "slot_fill", 2)]


def _f(v, d):
    return "–" if v is None else f"{v:.{d}f}"


def render_perf(before: dict, after: dict) -> str:
    b, a = dict(before["summary"]["all"]), after["summary"]["all"]
    # 베이스라인 UI는 모든 단계가 끝난 뒤 한 번에 표시 → 첫 일정 표시 = 전체 지연
    b.setdefault("first_plan_p50", None)
    b["first_plan_p50"] = b["first_plan_p50"] or b["latency_p50_all"]
    b["first_plan_p95"] = b.get("first_plan_p95") or b["latency_p95_all"]
    out = ["# 성능 최적화 전후 (E2E 20건, 같은 모델·같은 환경에서 연속 측정)", "",
           "| 지표 | 최적화 전 | 최적화 후 |", "|---|---|---|"]
    for label, key, d in LATENCY:
        bv, av = b.get(key), a.get(key)
        better = bv is not None and av is not None and av < bv
        out.append(f"| {label} | {_f(bv, d)} | {'**' + _f(av, d) + '**' if better else _f(av, d)} |")
    out += ["", "## 품질 유지 확인", "", "| 지표 | 최적화 전 | 최적화 후 |", "|---|---|---|"]
    out += [f"| {label} | {_f(b.get(k), d)} | {_f(a.get(k), d)} |" for label, k, d in QUALITY]
    out += ["", "## 단계별 평균 시간(초) — 병렬 단계는 겹쳐 실행", "", "| 전 | 후 |", "|---|---|",
            f"| {', '.join(f'{k} {v}' for k, v in b.get('node_time_mean', {}).items())} "
            f"| {', '.join(f'{k} {v}' for k, v in a.get('node_time_mean', {}).items())} |"]
    return "\n".join(out) + "\n"


def main(args):
    load = lambda p: json.loads(Path(p).read_text())
    md = render_perf(load(args.before), load(args.after))
    Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="eval/results/perf_baseline.json")
    ap.add_argument("--after", default="eval/results/perf_after.json")
    ap.add_argument("--out", default="docs/eval/perf.md")
    main(ap.parse_args())
