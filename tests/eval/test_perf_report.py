from eval.perf_report import render_perf

def summ(lat50, lat95, fp50, nodes, out_tok, success=1.0):
    return {"summary": {"all": {"latency_p50_all": lat50, "latency_p95_all": lat95, "first_plan_p50": fp50,
                                "first_plan_p95": fp50, "llm_time_s_all": 20.0, "search_time_s_all": 5.0,
                                "maps_time_s_all": 1.0, "output_tokens": out_tok, "input_tokens": 1000,
                                "success_rate": success, "region_acc": 1.0, "slot_fill": 1.0, "node_time_mean": nodes}}}


def test_render_perf_includes_latency_quality_and_nodes():
    b = summ(30.0, 58.0, None, {"compose": 20.0, "parse_request": 8.0}, 594)
    a = summ(22.0, 40.0, 9.0, {"brief": 12.0, "extract_preferences": 7.0}, 300)
    md = render_perf(b, a)
    assert "| 응답 지연 p50(초) | 30.0 | **22.0** |" in md
    assert "| 첫 일정 표시 p50(초) | 30.0 | **9.0** |" in md        # 베이스라인은 완료 시점에 한 번에 표시
    assert "| 일정 생성 성공률 | 1.00 | 1.00 |" in md
    assert "compose" in md and "brief" in md
