"""UI·PDF 텍스트 도우미 (순수 함수)."""
import re

_ROUTE = {"➡️": "→", "➡": "→"}   # "🚶 도보"의 🚶는 아래 이모지 제거로 지워짐
# 나눔고딕에 없는 그림 문자(이모지·기호·변형 선택자) — PDF에서 빈칸으로 찍히므로 제거
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")


def to_plain(text: str) -> str:
    """경로 화살표는 글자로 바꾸고 나머지 이모지는 지운다 (PDF 폰트용)."""
    for k, v in _ROUTE.items():
        text = text.replace(k, v)
    return re.sub(r"\s{2,}", " ", _EMOJI.sub("", text)).strip()


def highlight(text: str, keywords: list[str]) -> str:
    """리뷰 속 취향 키워드를 Streamlit 마크다운 배경색으로 강조."""
    for k in sorted({k for k in keywords if k}, key=len, reverse=True):
        text = text.replace(k, f":orange-background[{k}]")
    return text


def compact_route(detail: str | None) -> str:
    """경로 요약: 이모지를 글자로, 연속으로 반복되는 구간(도보 → 도보)은 하나로."""
    parts = [p.strip() for p in to_plain(detail or "").split("→") if p.strip()]
    merged = [p for n, p in enumerate(parts) if n == 0 or p != parts[n - 1]]
    return " → ".join(merged) or "이동"
