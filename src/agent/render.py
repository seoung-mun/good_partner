"""일정 → 사용자에게 보여줄 마크다운 (LLM 없이 결정적으로 렌더)."""
from src.domain.models import Briefing, ItineraryItem


def request_text(req: dict) -> str:
    lines = [f"- 동행: {req['group_type']}", f"- 스타일: {req['style']}"]
    if req.get("start_location"):
        lines.append(f"- 출발지: {req['start_location']}")
    if (req.get("detail") or "").strip():
        lines.append(f"- 상세 요청: {req['detail']}")
    return "\n".join(lines)


def render_markdown(req: dict, items: list[ItineraryItem], briefing: Briefing | None, weather: str) -> str:
    out = [f"## ✈️ {req['destination']} 맞춤 여행 계획표 ({req['days']}일)", ""]
    if briefing:
        out += [briefing.overview, "", f"🌤️ {briefing.weather_tip}", ""]
    elif weather:
        out += [f"🌤️ {weather.splitlines()[0]}", ""]
    day = None
    for it in items:
        if it.slot.day != day:
            day = it.slot.day
            out += ["", f"### 🗓️ Day {day}", "---"]
        if it.place is None:
            out.append(f"- ({it.slot.meal or it.slot.category}) 조건에 맞는 장소를 찾지 못했어요")
            continue
        if it.move_min is not None:
            out.append(f"   ⬇️ 이동: {it.move_detail or '이동'} (약 {it.move_min}분)")
        out.append(f"- **[{it.start}] {it.place.name}** ({it.place.category})")
        if it.reason:
            out.append(f"  > 💡 {it.reason}")
    return "\n".join(out)
