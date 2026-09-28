"""도메인 모델. 워크플로우 상태·tool 입출력·LLM 구조화 출력이 공유한다."""
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["식당", "카페", "관광지"]


class Slot(BaseModel):
    slot_id: str                                  # "d1-s2"
    day: int
    order: int
    category: Category
    meal: Literal["점심", "저녁"] | None = None


class Place(BaseModel):
    name: str
    region: str                                   # 메타데이터 '지역' ("부산광역시 중구")
    category: str
    address: str = ""                             # 상세 주소
    rating: float | None = None
    reviews: list[str] = []                       # 근거 리뷰


class ItineraryItem(BaseModel):
    slot: Slot
    place: Place | None = None
    start: str | None = None                      # "HH:MM"
    end: str | None = None
    move_min: int | None = None                   # 직전 장소에서 이동 시간
    move_detail: str | None = None
    reason: str = ""                              # 추천 사유 (브리핑)


class TripPreferences(BaseModel):
    keywords: list[str] = Field(description="리뷰 검색에 쓸 취향 키워드 1~4개 (예: 조용한, 바다뷰, 노포)")
    avoid: list[str] = Field(description="피하고 싶은 것 (없으면 빈 목록)")


class SlotReason(BaseModel):
    slot_id: str
    reason: str = Field(description="근거 리뷰에 기반한 1~2문장 추천 사유")


class Briefing(BaseModel):
    overview: str = Field(description="일정 전체를 소개하는 2~3문장")
    weather_tip: str = Field(description="날씨 데이터를 한 문장으로 요약한 조언")
    reasons: list[SlotReason]
