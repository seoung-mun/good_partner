"""하루 동선 지도 (pydeck — Streamlit 내장). 번호 핀 + 방문 순서 경로선."""
import math

import pydeck as pdk

from src.domain.models import ItineraryItem

PIN = [24, 95, 165]
PATH = [55, 138, 221]


def _zoom(span: float) -> float:
    """좌표 범위(도) → 모든 핀이 들어오는 줌 레벨 (웹 메르카토르: 줌 1 오를 때 보이는 범위 절반)."""
    return max(8.0, min(14.0, math.log2(360 / max(span, 0.005)) - 2.2))


def day_map(items: list[ItineraryItem]) -> pdk.Deck | None:
    """좌표가 있는 장소를 방문 순서대로 핀·선으로. 좌표가 하나도 없으면 None."""
    pts = [{"n": str(n), "name": it.place.name, "lat": it.place.lat, "lng": it.place.lng}
           for n, it in enumerate([i for i in items if i.place], start=1) if it.place.lat is not None]
    if not pts:
        return None
    lats, lngs = [p["lat"] for p in pts], [p["lng"] for p in pts]
    view = pdk.ViewState(latitude=(min(lats) + max(lats)) / 2, longitude=(min(lngs) + max(lngs)) / 2,
                         zoom=_zoom(max(max(lats) - min(lats), max(lngs) - min(lngs))))
    layers = [
        pdk.Layer("PathLayer", [{"path": [[p["lng"], p["lat"]] for p in pts]}], get_path="path",
                  get_color=PATH, width_min_pixels=3),
        pdk.Layer("ScatterplotLayer", pts, get_position="[lng, lat]", get_fill_color=PIN, radius_min_pixels=11,
                  pickable=True),
        pdk.Layer("TextLayer", pts, get_position="[lng, lat]", get_text="n", get_color=[255, 255, 255],
                  get_size=13, get_alignment_baseline="'center'"),
    ]
    return pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "{n}. {name}"},
                    map_style=pdk.map_styles.CARTO_LIGHT)
