from src.ui.text import highlight, to_plain


def test_to_plain_replaces_route_emoji_and_drops_others():
    assert to_plain("🚶 도보 ➡️ [버스] 17 ➡️ 🚶 도보") == "도보 → [버스] 17 → 도보"
    assert to_plain("🌤️ 맑음 ✈️") == "맑음"


def test_highlight_marks_keywords_only():
    assert highlight("해산물이 신선해요", ["해산물", "없는말"]) == ":orange-background[해산물]이 신선해요"
    assert highlight("그냥", []) == "그냥"


def test_compact_route_merges_repeated_walks():
    from src.ui.text import compact_route
    assert compact_route("🚶 도보 ➡️ 🚶 도보 ➡️ [지하철] 1호선 ➡️ 🚶 도보 ➡️ 🚶 도보") == "도보 → [지하철] 1호선 → 도보"
    assert compact_route(None) == "이동"
