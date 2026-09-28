"""날씨 tool (얇은 래퍼)."""
from langchain_core.tools import tool

from src.services.weather import forecast_summary


@tool
def get_weather_forecast(destination: str, dates: str) -> str:
    """
    도시명(destination)으로 위도/경도를 조회하고, 그 좌표로 5일 예보를 조회하여,
    사용자가 요청한 날짜(dates)의 날씨만 요약해 반환합니다. (3단계 날짜 파싱 적용)
    """
    return forecast_summary(destination, dates)
