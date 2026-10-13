from src.domain.models import ItineraryItem, Place, Slot
from src.ui.pdf import create_itinerary_pdf


def test_pdf_renders_places_with_multiple_reviews():
    item = ItineraryItem(slot=Slot(slot_id="d1-s1", day=1, order=1, category="식당"),
                         place=Place(name="국밥집", region="부산광역시 중구", category="식당",
                                     reviews=["국물이 진하고 맛있어요", "양이 많아요", "친절합니다"]),
                         start="12:00", end="13:30", reason="노포 리뷰가 많아요", move_min=15, move_detail="버스")
    pdf = create_itinerary_pdf([item], "부산 중구", "내일", "맑음", 1, "부산역")
    assert pdf.startswith(b"%PDF") and len(pdf) > 1000
