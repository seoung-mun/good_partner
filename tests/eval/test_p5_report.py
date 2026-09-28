from eval.p5_report import judge


def test_judge():
    assert judge(52, 60, "<=") == "달성" and judge(73, 60, "<=") == "미달"
    assert judge(0.9, 0.95, ">=") == "미달" and judge(None, 1, ">=") == "-"
