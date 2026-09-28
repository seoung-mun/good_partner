"""서비스 예외 계층. 모든 예외는 로그용 code와 사용자에게 보여줄 user_message를 가진다.

워크플로우는 이 예외를 던지지 않고 StepError(스키마)로 상태에 기록해 다음 단계를 계속 진행한다.
"""


class AppError(Exception):
    """서비스 예외의 기반 클래스."""

    code = "APP_ERROR"
    user_message = "요청을 처리하는 중 문제가 발생했어요. 잠시 후 다시 시도해 주세요."

    def __init__(self, detail: str = ""):
        """예외 상세를 받아 사용자 안내 문구를 구성."""
        super().__init__(detail or self.user_message)
        self.detail = detail


class RegionNotFoundError(AppError):
    """목적지를 행정구역(시/도, 시/군/구)으로 확정하지 못함."""

    code = "REGION_NOT_FOUND"

    def __init__(self, destination: str):
        """예외 상세를 받아 사용자 안내 문구를 구성."""
        super().__init__(f"목적지 해석 실패: {destination}")
        self.user_message = (f"'{destination}'이(가) 어느 지역인지 확인하지 못했어요. "
                             "'부산 중구'처럼 시/도와 시/군/구를 함께 적어 주세요.")


class VectorStoreUnavailableError(AppError):
    """장소 검색용 벡터 인덱스를 불러오지 못함."""

    code = "VECTOR_STORE_UNAVAILABLE"
    user_message = "장소 데이터베이스를 불러오지 못했어요. 관리자에게 문의해 주세요."


class LLMOutputError(AppError):
    """LLM 호출 실패 또는 구조화 출력(스키마) 파싱 실패."""

    code = "LLM_OUTPUT_INVALID"

    def __init__(self, step: str, cause: Exception):
        """예외 상세를 받아 사용자 안내 문구를 구성."""
        super().__init__(f"{step}: {type(cause).__name__}: {cause}")
        self.step = step
        self.user_message = "AI 응답을 해석하지 못해 일부 설명을 생략했어요. 일정은 그대로 이용할 수 있어요."


class ExternalServiceError(AppError):
    """외부 API(Google Maps, OpenWeatherMap) 호출 실패."""

    code = "EXTERNAL_SERVICE_FAILED"

    def __init__(self, service: str, detail: str = ""):
        """예외 상세를 받아 사용자 안내 문구를 구성."""
        super().__init__(f"{service}: {detail}")
        self.service = service
        self.user_message = f"{service} 정보를 가져오지 못해 일부 정보가 빠졌어요."
