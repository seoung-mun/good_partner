"""도메인 모델. 워크플로우 상태·tool 입출력·LLM 구조화 출력이 공유한다."""
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["식당", "카페", "관광지"]


class Slot(BaseModel):
    """일정의 빈칸 하나 (일차·순서·카테고리·식사 구분)."""
    slot_id: str                                  # "d1-s2"
    day: int
    order: int
    category: Category
    meal: Literal["아침", "점심", "저녁"] | None = None
    when: Literal["밤"] | None = None             # 시간대 슬롯 (밤 관광: 저녁 식사 뒤, 밤에 가야 좋은 곳만)


class Place(BaseModel):
    """검색된 장소 (메타데이터 기반, 근거 리뷰 포함)."""
    name: str
    region: str                                   # 메타데이터 '지역' ("부산광역시 중구")
    category: str
    address: str = ""                             # 상세 주소
    rating: float | None = None
    reviews: list[str] = []                       # 근거 리뷰 (쿼리와 가까운 순)
    n_reviews: int = 0                            # 수집된 전체 리뷰 수
    quote: str = ""                               # 카드에 인용할 리뷰 한 줄 (취향 키워드 포함 우선)
    score: float = 0.0                            # 검색 순위 점수 (RRF)
    time_fit: dict[str, float] = {}               # 시간대별 적합도: 그 시간대 쪽으로 기운 리뷰 비율 (예: {"밤": 0.4})
    lat: float | None = None                      # 좌표 (지오코딩 후)
    lng: float | None = None


class ItineraryItem(BaseModel):
    """슬롯 + 배정된 장소 + 시간표 + 추천 사유."""
    slot: Slot
    place: Place | None = None
    start: str | None = None                      # "HH:MM"
    end: str | None = None
    move_min: int | None = None                   # 직전 장소(from_lodging이면 숙소·도착지)에서 이동 시간
    from_lodging: bool = False                    # 하루 첫 장소: 숙소(첫날은 도착지)에서 출발
    move_detail: str | None = None
    reason: str = ""                              # 추천 사유 (브리핑)
    alternatives: list[Place] = []                # 교체 후보 (같은 권역 차순위)
    lodging_min: int | None = None                # 그날 마지막 장소 → 숙소 복귀 이동 시간 (마지막 날 제외)
    lodging_detail: str | None = None


class TripPreferences(BaseModel):
    """취향 키워드 (검색 쿼리에 사용). 카테고리별 목록이 있으면 그것만, 모두 비었으면 공통 목록을 쓴다."""
    # maxItems: 구조화 출력 문법이 목록 길이를 강제 → 소형 모델이 키워드를 끝없이 생성하는 것을 막음
    keywords: list[str] = Field(max_length=4, description="리뷰 검색에 쓸 취향 키워드 1~4개 (예: 조용한, 바다뷰, 노포)")
    avoid: list[str] = Field(max_length=4, description="피하고 싶은 것 (없으면 빈 목록)")
    food: list[str] = Field(default_factory=list, max_length=3)
    cafe: list[str] = Field(default_factory=list, max_length=3)
    sight: list[str] = Field(default_factory=list, max_length=3)

    def by_category(self) -> dict[str, list[str]]:
        """카테고리 → 검색 키워드. '흑돼지'가 카페 검색에 섞이지 않게 카테고리별 목록을 우선."""
        if self.food or self.cafe or self.sight:
            return {"식당": self.food, "카페": self.cafe, "관광지": self.sight}
        return {"식당": self.keywords, "카페": self.keywords, "관광지": self.keywords}


class SlotReason(BaseModel):
    """슬롯별 추천 사유."""
    slot_id: str
    reason: str = Field(description="근거 리뷰에 기반한 추천 사유 1문장(60자 이내)")


class Briefing(BaseModel):
    """LLM 브리핑: 개요, 날씨 팁, 슬롯별 사유."""
    overview: str = Field(description="일정 전체를 소개하는 1~2문장")
    weather_tip: str = Field(description="날씨 데이터를 한 문장으로 요약한 조언")
    reasons: list[SlotReason]


class StepError(BaseModel):
    """워크플로우 단계에서 발생한 오류. 예외를 던지는 대신 상태에 쌓아 서비스를 계속 진행한다."""

    step: str                                     # 발생 단계 (parse_request, search, ...)
    code: str                                     # AppError.code
    message: str                                  # 사용자에게 보여줄 안내
    detail: str = ""                              # 로그·디버깅용 원인

    @classmethod
    def from_error(cls, step: str, err: Exception) -> "StepError":
        """AppError면 code·user_message를, 그 외 예외는 일반 오류로 변환."""
        return cls(step=step, code=getattr(err, "code", "APP_ERROR"),
                   message=getattr(err, "user_message", "요청을 처리하는 중 문제가 발생했어요."),
                   detail=getattr(err, "detail", "") or f"{type(err).__name__}: {err}")
