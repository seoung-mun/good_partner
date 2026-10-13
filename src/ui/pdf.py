"""일정 PDF. ItineraryItem을 직접 읽는다 (중간 dict 변환 없음)."""
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from src.domain.models import ItineraryItem
from src.ui.text import to_plain

ROOT = Path(__file__).resolve().parents[2]
FONT = ROOT / "NanumGothic.ttf"
FONT_BOLD = ROOT / "NanumGothicBold.ttf"


def _line(pdf: FPDF, h: float, text: str, size: int, style: str = "", indent: float | None = None) -> None:
    """한 줄(넘치면 줄바꿈)을 쓰고 왼쪽 여백으로 돌아간다."""
    pdf.set_font("NanumGothic", style, size)
    if indent is not None:
        pdf.set_x(indent)
    pdf.multi_cell(0, h, text=text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def create_itinerary_pdf(items: list[ItineraryItem], destination: str, dates: str, weather: str,
                         total_days: int, start_location: str | None = None) -> bytes:
    """일차별 일정표 PDF 바이트. 한글 폰트 파일이 없으면 FileNotFoundError."""
    for f in (FONT, FONT_BOLD):
        if not f.exists():
            raise FileNotFoundError(f"PDF 한글 폰트가 없습니다: {f}")
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("NanumGothic", "", str(FONT))
    pdf.add_font("NanumGothic", "B", str(FONT_BOLD))

    pdf.set_font("NanumGothic", "B", 24)
    pdf.cell(0, 20, text=f"{destination} 여행 계획", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("NanumGothic", "", 12)
    pdf.cell(0, 10, text=f"기간: {dates}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    if start_location:
        pdf.cell(0, 8, text=f"출발지/숙소: {start_location}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    if weather.strip():
        pdf.set_font_size(10)
        pdf.multi_cell(0, 5, text=f"날씨: {to_plain(weather)}", align="C")
    pdf.ln(10)

    for day in range(1, total_days + 1):
        if day > 1:
            pdf.ln(15)
        _line(pdf, 15, f"Day {day}", 18, "B")
        today = [it for it in items if it.slot.day == day and it.place is not None]
        if not today:
            _line(pdf, 10, "  - 계획된 일정이 없습니다.", 11)
        for it in today:
            if it.move_min is not None and not it.from_lodging:
                pdf.set_text_color(100, 100, 100)
                _line(pdf, 8, f"      |  약 {it.move_min}분 : {to_plain(it.move_detail or '이동')}", 10)
                pdf.set_text_color(0, 0, 0)
            when = f"[{it.start}-{it.end or ''}]" if it.start else "[시간 미정]"
            _line(pdf, 8, f"  ● {when} {to_plain(it.place.name)} ({it.place.category})", 12, "B")
            if it.reason:
                _line(pdf, 5, to_plain(it.reason), 10, indent=20)
            for review in it.place.reviews:
                _line(pdf, 4, f"  • {to_plain(review)}", 9, indent=20)
            pdf.ln(2)
        pdf.ln(10)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(5)
        _line(pdf, 10, "메모:", 14)
        pdf.ln(20)
    return bytes(pdf.output())
