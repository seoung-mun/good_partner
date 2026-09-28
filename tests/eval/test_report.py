from eval.report import render

R = {"llm_model": "m", "k": 5, "summary": {"all": {"n": 2, "region_acc": 0.5, "category_acc": 1.0,
      "region_hit1": 0.5, "diversity": 1.0, "empty": 0.0, "llm_calls": 3.0, "latency_s": 1.0},
      "gwangju": {"n": 1, "region_acc": 0.0, "category_acc": 1.0, "region_hit1": 0.0,
                  "diversity": 1.0, "empty": 0.0, "llm_calls": 3.0, "latency_s": 1.0}}}
E = {"llm_model": "m", "summary": {"all": {"n": 1, "ok": 1.0, "slot_fill": 0.8, "category_adherence": 0.6,
      "region_acc": 0.9, "dup_rate": 0.0, "overflow": 0.0, "total_move_min": 50.0, "llm_calls": 31.0,
      "input_tokens": 1000.0, "output_tokens": 100.0, "maps_calls": 12.0, "latency_s": 90.0, "n_places": 5.0}}}
J = {"judge_model": "j", "mean_score": 3.5, "mean_trip_score": 3.4,
     "status_counts": {"ok": 4, "unmatched": 1, "parse_error": 0}}


def test_render_contains_key_numbers():
    md = render(R, E, J, {"n": 40, "quadratic_weighted_kappa": 0.71})
    assert "| gwangju | 1 | 0.00 |" in md
    assert "31.0" in md and "3.50" in md and "0.71" in md


def test_render_without_agreement():
    assert "수기 라벨 미완료" in render(R, E, J, None)
