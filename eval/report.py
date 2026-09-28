import argparse
import json
from pathlib import Path


def _f(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def render(retrieval: dict, e2e: dict, judge: dict, agreement: dict | None) -> str:
    out = ["# 베이스라인 평가 결과", "",
           f"- 생성 모델: `{e2e['llm_model']}` / judge: `{judge['judge_model']}`", ""]
    out += ["## A. 검색 컴포넌트 (k={})".format(retrieval["k"]), "",
            "| 그룹 | n | 지역 정확도@k | 지역 Hit@1 | 카테고리 정확도@k | 다양성@k | 빈 결과 비율 | LLM 호출 |",
            "|---|---|---|---|---|---|---|---|"]
    for g, s in retrieval["summary"].items():
        out.append(f"| {g} | {s['n']} | {_f(s['region_acc'])} | {_f(s['region_hit1'])} | "
                   f"{_f(s['category_acc'])} | {_f(s['diversity'])} | {_f(s['empty'])} | {_f(s['llm_calls'], 1)} |")
    s = e2e["summary"]["all"]
    out += ["", "## B. E2E 일정 생성", "", "| 지표 | 값 |", "|---|---|"]
    for label, key, d in [("성공률", "ok", 2), ("슬롯 충족률", "slot_fill", 2),
                          ("카테고리 준수율", "category_adherence", 2), ("일정 지역 정확도", "region_acc", 2),
                          ("중복률", "dup_rate", 2), ("초과 배정 수", "overflow", 1),
                          ("총 이동 시간(분)", "total_move_min", 1), ("LLM 호출 수", "llm_calls", 1),
                          ("입력 토큰", "input_tokens", 0), ("출력 토큰", "output_tokens", 0),
                          ("Maps 호출 수", "maps_calls", 1), ("소요 시간(초)", "latency_s", 1)]:
        out.append(f"| {label} | {_f(s.get(key), d)} |")
    c = judge["status_counts"]
    out += ["", "## C. 취향 적합도 (LLM-as-judge, 1~5)", "",
            f"- 장소 평균: **{_f(judge['mean_score'])}**, 일정 평균: {_f(judge['mean_trip_score'])}",
            f"- 채점 {c['ok']}건 / 메타데이터 불일치 {c['unmatched']}건 / 파싱 실패 {c['parse_error']}건"]
    if agreement:
        out.append(f"- judge 신뢰도: 수기 라벨 {agreement['n']}건과 가중 kappa **{_f(agreement['quadratic_weighted_kappa'])}**")
    else:
        out.append("- judge 신뢰도: 수기 라벨 미완료")
    return "\n".join(out) + "\n"


def main(args):
    load = lambda p: json.loads(Path(p).read_text())
    agreement = load(args.agreement) if args.agreement and Path(args.agreement).exists() else None
    md = render(load(f"eval/results/retrieval_{args.tag}.json"), load(f"eval/results/e2e_{args.tag}.json"),
                load(f"eval/results/judge_{args.tag}.json"), agreement)
    Path(f"docs/eval/{args.tag}.md").parent.mkdir(parents=True, exist_ok=True)
    Path(f"docs/eval/{args.tag}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--agreement", default="eval/results/agreement_baseline.json")
    main(ap.parse_args())
