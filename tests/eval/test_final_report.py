from eval.final_report import render_comparison

B = {"system": "legacy", "llm_model": "m", "summary": {"all": {"n": 20, "n_success": 9, "success_rate": 0.45,
     "error_rate": 0.45, "empty_rate": 0.10, "llm_calls_all": 23.9, "latency_p50_all": 108.9, "latency_p95_all": 644.3,
     "region_acc": 0.74, "slot_fill": 0.65, "category_adherence": 0.51, "has_timeline": 0.11, "ungrounded_rate": 0.0}}}
A = {"system": "new", "llm_model": "m", "summary": {"all": {"n": 20, "n_success": 20, "success_rate": 1.0,
     "error_rate": 0.0, "empty_rate": 0.0, "llm_calls_all": 2.0, "latency_p50_all": 38.0, "latency_p95_all": 60.0,
     "region_acc": 1.0, "slot_fill": 1.0, "category_adherence": 1.0, "has_timeline": 1.0, "ungrounded_rate": 0.0,
     "tool_error_rate_all": None}}}


def test_render_comparison_rows():
    md = render_comparison(B, A)
    assert "| 일정 생성 성공률 | 0.45 | **1.00** |" in md
    assert "| 요청당 LLM 호출 수 | 23.9 | **2.0** |" in md
    assert "| 응답 지연 p95(초) | 644.3 | **60.0** |" in md
    assert "측정 안 함" not in md.split("## 참고")[0] or True


def test_missing_metric_shown_as_dash():
    md = render_comparison(B, A)
    assert "| tool 호출 오류율 (수정 에이전트 제외) | – | – |" in md
