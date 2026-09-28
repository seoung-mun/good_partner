from eval.compare import diff_rows


def test_diff_rows_detects_changed_topk():
    a = {"rows": [{"id": "x", "top": [1]}, {"id": "y", "top": [2]}]}
    b = {"rows": [{"id": "x", "top": [1]}, {"id": "y", "top": [3]}]}
    assert diff_rows(a, b) == ["y"]
