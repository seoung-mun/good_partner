import ast

from src.agent.workflow import to_legacy_itinerary
from src.domain.models import ItineraryItem, Place, Slot


def _page_functions():
    """streamlit 페이지 전체를 실행하지 않고 함수 정의만 로드."""
    tree = ast.parse(open("pages/trip_planner.py").read())
    body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), "trip_planner", "exec"), ns)
    return ns


def test_pdf_renders_places_with_multiple_reviews():
    item = ItineraryItem(slot=Slot(slot_id="d1-s1", day=1, order=1, category="식당"),
                         place=Place(name="국밥집", region="부산광역시 중구", category="식당",
                                     reviews=["국물이 진하고 맛있어요", "양이 많아요", "친절합니다"]),
                         start="12:00", end="13:30", reason="노포 리뷰가 많아요")
    pdf = _page_functions()["create_itinerary_pdf"](to_legacy_itinerary([item]), "부산 중구", "내일", "맑음", "", 1, "부산역")
    assert pdf and len(pdf) > 1000
