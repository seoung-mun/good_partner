"""일정 카드 렌더링 (Streamlit). 상태 변경은 하지 않고, 사용자가 고른 교체 후보만 돌려준다."""
import streamlit as st

from src.domain.models import ItineraryItem, Place
from src.ui.text import compact_route, highlight

BADGE = {"식당": "orange", "카페": "green", "관광지": "blue"}
LONG_MOVE_MIN = 60
MAX_REVIEWS_SHOWN = 30


def chips(labels: list[str], keywords: list[str]) -> None:
    """입력 요약 칩(회색) + 추출된 취향 키워드 칩(파랑)."""
    st.markdown(" ".join([f":gray-badge[{x}]" for x in labels if x] + [f":blue-badge[{k}]" for k in keywords]))


def move_line(minutes: int, detail: str | None, label: str = "") -> None:
    """카드 사이 이동 줄. 60분을 넘으면 경고."""
    text = f"{label}{compact_route(detail)} · {minutes}분"
    if minutes > LONG_MOVE_MIN:
        st.warning(f"{text} — 이동이 길어요", icon=":material/directions_bus:")
    else:
        st.caption(f":material/directions_bus: {text}")


def place_card(n: int, it: ItineraryItem, keywords: list[str], all_reviews: list[str],
               taken: set[tuple[str, str]]) -> Place | None:
    """장소 카드. '다른 곳으로 바꾸기'에서 대안을 고르면 그 Place를 반환."""
    p, sid = it.place, it.slot.slot_id
    with st.container(border=True):
        head, badge = st.columns([4, 1])
        head.markdown(f"**{n}. {p.name}**  \n:gray[{it.start} – {it.end}]")
        badge.markdown(f":{BADGE.get(p.category, 'gray')}-badge[{it.slot.meal or p.category}]")
        meta = [f"★ {p.rating:.1f}" if p.rating else "", f"리뷰 {p.n_reviews}개" if p.n_reviews else ""]
        st.caption(" · ".join(x for x in meta if x))
        if p.quote:
            st.markdown(f"> {highlight(p.quote[:140], keywords)}")
        if it.reason:
            st.caption(f":material/lightbulb: {it.reason}")
        alts = [a for a in it.alternatives if (a.name, a.region) not in taken]
        with st.popover("다른 곳으로 바꾸기", disabled=not alts):
            for k, a in enumerate(alts):
                label = f"{a.name}" + (f" · ★ {a.rating:.1f}" if a.rating else "")
                if st.button(label, key=f"swap-{sid}-{k}", use_container_width=True):
                    return a
        with st.expander(f"리뷰 {len(all_reviews)}개 보기"):
            for r in all_reviews[:MAX_REVIEWS_SHOWN]:
                st.markdown(f"- {highlight(r, keywords)}")
            if len(all_reviews) > MAX_REVIEWS_SHOWN:
                st.caption(f"외 {len(all_reviews) - MAX_REVIEWS_SHOWN}개")
    return None
