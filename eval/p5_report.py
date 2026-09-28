"""P5 결과 리포트: 목표 대비 달성 여부 표 → docs/eval/p5.md.

입력: 검색 단계(search_*.json), E2E 전(perf_after.json)·후(p5_after.json), 튜닝 이력(p5_after_r*.json), 수정 에이전트(edit_p5.json).
"""
import json
from pathlib import Path

from eval.metrics import route_metrics

R = Path("eval/results")
OUT = Path("docs/eval/p5.md")
ABLATION = [("1_current", "① 기존: 전국 top-50 → 지역 사후 필터, 청크 1개 점수"),
            ("2_prefilter", "② 지역·카테고리 선필터 + 장소 단위 점수(리뷰 3개 평균)"),
            ("3a_rrf_oldquery", "RRF(취향·기본·인기), 기존 쿼리"),
            ("3b_rrf_query", "RRF(취향·기본·인기), 리뷰 말투 쿼리"),
            ("3e_newquery_prefpop", "RRF(취향·인기), 리뷰 말투 쿼리"),
            ("3i_oldquery_preffirst", "취향 우선 순서, 기존 쿼리 (= ② 재현 확인)"),
            ("3_final", "③ 최종: 선필터 + 장소 점수 + 리뷰 말투 쿼리, 취향 우선")]


def judge(value, target: float, op: str) -> str:
    """목표 판정: op는 '<=' 또는 '>='. 값이 없으면 '-'."""
    if value is None:
        return "-"
    ok = value <= target if op == "<=" else value >= target
    return "달성" if ok else "미달"


def _load(name: str) -> dict | None:
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render() -> str:
    lines = ["# P5 검색 품질 · 동선 결과 (목표 대비)", "",
             "모델 qwen2.5:7b(Ollama), 같은 머신. E2E는 캐시 끔(Maps·지오코딩 포함 첫 호출 비용 기준).", ""]
    # --- 검색
    base = _load("search_1_current")["summary"]["rank"]["all"]
    fin = _load("search_3_final")
    lines += ["## 1. 검색 (실버 골든셋 131건 + 지역 골든셋 73건)", "",
              "| 구성 | MRR@10 | nDCG@10 | Recall@20 | 지역 정확도 |", "|---|---|---|---|---|"]
    for key, label in ABLATION:
        d = _load(f"search_{key}")
        if d:
            r, g = d["summary"]["rank"]["all"], d["summary"]["region"]["all"]
            lines.append(f"| {label} | {_f(r['mrr@10'])} | {_f(r['ndcg@10'])} | {_f(r['recall@20'])} | {_f(g['region_acc'])} |")
    fr, fg = fin["summary"]["rank"]["all"], fin["summary"]["region"]["all"]
    lines += ["", "| 목표 | 기준 ① | 최종 ③ | 판정 |", "|---|---|---|---|",
              f"| MRR@10 ≥ ①×1.2 | {_f(base['mrr@10'])} | {_f(fr['mrr@10'])} | {judge(fr['mrr@10'], base['mrr@10'] * 1.2, '>=')} |",
              f"| nDCG@10 > ① | {_f(base['ndcg@10'])} | {_f(fr['ndcg@10'])} | {judge(fr['ndcg@10'], base['ndcg@10'] + 1e-9, '>=')} |",
              f"| Recall@20 > ① | {_f(base['recall@20'])} | {_f(fr['recall@20'])} | {judge(fr['recall@20'], base['recall@20'] + 1e-9, '>=')} |",
              f"| 빈 결과율 = 0 | - | {_f(fg['empty'])} | {judge(fg['empty'], 0, '<=')} |", ""]
    # --- 동선·가드레일
    before, after = _load("perf_after"), _load("p5_after")
    rb, ra = route_metrics(before["rows"]), after["summary"]["route"]
    sa = after["summary"]["all"]
    sb = before["summary"]["all"]
    rows = [("구간 이동 p50(분)", rb["leg_p50"], ra["leg_p50"], 25, "<="),
            ("구간 이동 p95(분)", rb["leg_p95"], ra["leg_p95"], 60, "<="),
            ("60분 초과 구간 비율", rb["leg_over60_rate"], ra["leg_over60_rate"], 0.10, "<="),
            ("저녁 20시 이후 시작 비율", rb["late_dinner_rate"], ra["late_dinner_rate"], 0.10, "<="),
            ("여행당 총 이동(분)", rb["move_per_trip"], ra["move_per_trip"], None, ""),
            ("성공률", sb["success_rate"], sa["success_rate"], 1.0, ">="),
            ("지역 정확도", sb["region_acc"], sa["region_acc"], 1.0, ">="),
            ("환각 장소 비율", sb["ungrounded_rate"], sa["ungrounded_rate"], 0.0, "<="),
            ("슬롯 충족률", sb["slot_fill"], sa["slot_fill"], 0.95, ">="),
            ("지연 p50(초)", sb["latency_p50_all"], sa["latency_p50_all"], 25, "<="),
            ("첫 일정 표시 p95(초)", sb.get("first_plan_p95"), sa.get("first_plan_p95"), 15, "<=")]
    lines += ["## 2. 동선 · 가드레일 (E2E 20건, 대중교통)", "",
              "| 지표 | 전 (P4) | 후 (P5) | 목표 | 판정 |", "|---|---|---|---|---|"]
    for name, b, a, t, op in rows:
        lines.append(f"| {name} | {_f(b, 2)} | {_f(a, 2)} | {'' if t is None else op + ' ' + str(t)} | "
                     f"{'-' if t is None else judge(a, t, op)} |")
    lines += ["", "전(P4)은 슬롯 템플릿이 달라(마지막 날 관광 1곳) 슬롯 충족률은 각자의 템플릿 기준.", ""]
    # --- 튜닝 이력
    hist = [(n, _load(f"p5_after_{n}")) for n in ("r0", "r1")]
    lines += ["## 3. 튜닝 이력", "", "| 회차 | 설정 | 구간 p50 | 구간 p95 | 60분 초과 | 늦은 저녁 |", "|---|---|---|---|---|---|"]
    notes = {"r0": "좌표 후보 25, 대중교통 반경 8km", "r1": "좌표 후보 50, 반경 5km"}
    for n, d in hist:
        if d:
            r = d["summary"]["route"]
            lines.append(f"| {n} | {notes[n]} | {r['leg_p50']} | {r['leg_p95']} | {_f(r['leg_over60_rate'], 2)} | "
                         f"{_f(r['late_dinner_rate'], 2)} |")
    lines.append(f"| 최종 | r1 설정 + 새 템플릿(아침·숙소 복귀) | {ra['leg_p50']} | {ra['leg_p95']} | "
                 f"{_f(ra['leg_over60_rate'], 2)} | {_f(ra['late_dinner_rate'], 2)} |")
    # --- 수정 에이전트
    ed = _load("edit_p5")
    if ed:
        s = ed["summary"]["all"]
        lines += ["", "## 4. 수정 에이전트 회귀", "", f"성공률 {_f(s.get('success'), 2)} (목표 ≥ 0.95) — "
                  f"{judge(s.get('success'), 0.95, '>=')}, tool 실패율 {_f(s.get('tool_failure_rate'), 2)}, 잘못된 tool 호출률 {_f(s.get('invalid_tool_call_rate'), 2)}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.write_text(render())
    print(OUT.read_text())
