"""베이스라인(legacy) vs 리팩토링 후(new) E2E 핵심 지표 비교 → docs/eval/final.md"""
import argparse
import json
from pathlib import Path

ROWS = [  # (라벨, 키, 소수 자릿수, 높을수록 좋은가)
    ("일정 생성 성공률", "success_rate", 2, True),
    ("오류(무한 루프 등) 비율", "error_rate", 2, False),
    ("빈 일정 비율", "empty_rate", 2, False),
    ("요청당 LLM 호출 수", "llm_calls_all", 1, False),
    ("응답 지연 p50(초)", "latency_p50_all", 1, False),
    ("응답 지연 p95(초)", "latency_p95_all", 1, False),
    ("일정 지역 정확도", "region_acc", 2, True),
    ("슬롯 충족률", "slot_fill", 2, True),
    ("카테고리 준수율", "category_adherence", 2, True),
    ("타임라인 완성률", "has_timeline", 2, True),
    ("환각 장소율 (DB에 없는 장소)", "ungrounded_rate", 2, False),
    ("tool 호출 오류율 (수정 에이전트 제외)", "tool_error_rate_all", 2, False),
]


def _f(v, d):
    return "–" if v is None else f"{v:.{d}f}"


def render_comparison(before: dict, after: dict) -> str:
    b, a = before["summary"]["all"], after["summary"]["all"]
    out = ["# 리팩토링 전후 비교 (E2E, 여행 요청 20건)", "",
           f"- 모델: `{after['llm_model']}` (전후 동일)", "- 성공 = 예외 없이 장소 1개 이상. 품질 지표는 성공한 요청 기준 평균, 성공률·지연·호출 수는 전체 기준.",
           "", "| 지표 | 리팩토링 전 | 리팩토링 후 |", "|---|---|---|"]
    for label, key, d, higher in ROWS:
        bv, av = b.get(key), a.get(key)
        better = bv is not None and av is not None and (av > bv if higher else av < bv)
        out.append(f"| {label} | {_f(bv, d)} | {'**' + _f(av, d) + '**' if better else _f(av, d)} |")
    return "\n".join(out) + "\n"


def main(args):
    load = lambda p: json.loads(Path(p).read_text())
    md = render_comparison(load(args.before), load(args.after))
    Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="eval/results/e2e_baseline.json")
    ap.add_argument("--after", default="eval/results/e2e_final.json")
    ap.add_argument("--out", default="docs/eval/final.md")
    main(ap.parse_args())
