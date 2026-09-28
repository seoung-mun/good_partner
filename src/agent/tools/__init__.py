"""LLM에 노출되는 tool 목록. tool 본문은 얇게 두고 로직은 domain/retrieval/services에 둔다."""
from src.agent.tools.itinerary import (confirm_and_download_pdf, optimize_and_get_routes,
                                       plan_itinerary_timeline)
from src.agent.tools.place import delete_place, find_and_select_best_place, replace_place
from src.agent.tools.weather import get_weather_forecast

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
