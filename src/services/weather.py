"""OpenWeatherMap 예보 조회·요약."""
import datetime
import os

import requests

from src.logging_config import get_logger

log = get_logger(__name__)


def _google_client():
    """Google Maps 클라이언트 (지오코딩 fallback). 테스트에서 교체 가능하도록 함수로 감쌈."""
    from src.config import GMAPS_CLIENT
    return GMAPS_CLIENT


def forecast_summary(destination: str, dates: str) -> str:
    """
    도시명(destination)으로 위도/경도를 조회하고, 그 좌표로 5일 예보를 조회하여,
    사용자가 요청한 날짜(dates)의 날씨만 요약해 반환합니다. (3단계 날짜 파싱 적용)
    """
    API_KEY = os.getenv("OWM_API_KEY")
    if not API_KEY:
        return "오류: OWM_API_KEY가 .env 파일에 설정되지 않았습니다."

    geo_url = "https://api.openweathermap.org/geo/1.0/direct"
    geo_params = {'q': f"{destination},KR", 'limit': 1, 'appid': API_KEY}
    lat, lon = None, None
    try:
        response = requests.get(geo_url, params=geo_params, timeout=5)
        response.raise_for_status()
        geo_data = response.json()
        if geo_data:
            lat = geo_data[0]['lat']
            lon = geo_data[0]['lon']
    except Exception as e:
        log.warning("OWM 지오코딩 실패: %s", e)
    if lat is None:  # OWM은 '강릉', '부산 중구' 같은 한국 지명을 자주 못 찾음 → Google Geocoding 좌표
        try:
            res = _google_client().geocode(destination, language="ko")
            if res:
                loc = res[0]["geometry"]["location"]
                lat, lon = loc["lat"], loc["lng"]
        except Exception as e:
            log.warning("Google 지오코딩 실패: %s", e)
    if lat is None:
        return f"오류: '{destination}'의 좌표(Geocoding)를 찾을 수 없습니다."

    forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
    forecast_params = {'lat': lat, 'lon': lon, 'appid': API_KEY, 'units': 'metric', 'lang': 'kr'}
    forecasts = None
    try:
        response = requests.get(forecast_url, params=forecast_params, timeout=10)
        response.raise_for_status()
        forecast_data = response.json()
        forecasts = forecast_data.get('list', [])
    except Exception as e:
        return f"오류: Forecast API 호출 중 문제 발생: {e}"
    if not forecasts:
        return "오류: Forecast API에서 'list' 데이터를 찾을 수 없습니다."

    target_date_str = ""
    today = datetime.datetime.now()
    
    try:
        target_date_obj = datetime.datetime.strptime(dates, "%Y년 %m월 %d일")
        target_date_str = target_date_obj.strftime("%Y-%m-%d")
    except ValueError:
        try:
            target_date_obj = datetime.datetime.strptime(dates, "%Y년%m월%d일")
            target_date_str = target_date_obj.strftime("%Y-%m-%d")
        except ValueError:
            try:
                target_date_obj = datetime.datetime.strptime(dates, "%m월 %d일")
                target_date_obj = target_date_obj.replace(year=today.year)
                target_date_str = target_date_obj.strftime("%Y-%m-%d")
            except ValueError:
                 if "주말" in dates or "토요일" in dates:
                     days_until_saturday = (5 - today.weekday() + 7) % 7
                     saturday = today + datetime.timedelta(days=days_until_saturday)
                     target_date_str = saturday.strftime("%Y-%m-%d")
                 elif "내일" in dates:
                     tomorrow = today + datetime.timedelta(days=1)
                     target_date_str = tomorrow.strftime("%Y-%m-%d")
                 else: 
                     tomorrow = today + datetime.timedelta(days=1)
                     target_date_str = tomorrow.strftime("%Y-%m-%d")
    
    output_str = f"[{destination} ({target_date_str}) 날씨 예보 (OWM)]\n"
    found = False
    for forecast in forecasts:
        if forecast['dt_txt'].startswith(target_date_str):
            time_utc = forecast['dt_txt'].split(' ')[1][:5]
            temp = forecast['main']['temp'] 
            desc = forecast['weather'][0]['description']
            output_str += f"- {time_utc} (UTC): {temp:.1f}℃, {desc}\n"
            found = True
    
    if not found:
        return f"정보: {target_date_str} 날짜의 예보를 찾을 수 없습니다. (OWM은 5일치만 제공)"
    
    return output_str
