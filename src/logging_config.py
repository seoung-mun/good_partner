"""전역 로깅 설정. 모든 모듈은 get_logger(__name__)로 로거를 얻고, 레벨은 환경변수 LOG_LEVEL로 조절한다."""
import logging
import os

_ROOT = "castone"
_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def setup_logging(force: bool = False) -> None:
    """루트 로거(castone)에 핸들러를 한 번만 붙인다. force=True면 레벨·핸들러를 다시 설정."""
    root = logging.getLogger(_ROOT)
    if root.handlers and not force:
        return
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """모듈 로거. 'castone.<모듈 경로>' 계층으로 만들어 전역 설정을 상속한다."""
    setup_logging()
    return logging.getLogger(f"{_ROOT}.{name}")
