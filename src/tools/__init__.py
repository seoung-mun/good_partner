"""tool 패키지. 기존 `from src.tools import ...` 호환을 위해 공개 이름을 다시 내보낸다."""
from src.config import GMAPS_CLIENT
from src.tools.chains import desc_chain, query_gen_chain, region_chain
from src.tools.geo import (calculate_distance_time, calculate_haversine_distance, get_coordinates,
                           get_detailed_route, resolve_admin_region)
from src.tools.itinerary_tools import (_solve_tsp, confirm_and_download_pdf, optimize_and_get_routes,
                                       plan_itinerary_timeline)
from src.tools.place_tools import delete_place, find_and_select_best_place, replace_place
from src.tools.retrieval import _extract_reviews, _filter_candidates, _search_docs, retrieve_candidates
from src.tools.weather import get_weather_forecast

TOOLS = [
    find_and_select_best_place,
    plan_itinerary_timeline,
    optimize_and_get_routes,
    get_weather_forecast,
    delete_place,
    replace_place,
    confirm_and_download_pdf
]
AVAILABLE_TOOLS = {tool.name: tool for tool in TOOLS}
