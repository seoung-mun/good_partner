import logging

from src.errors import (AppError, ExternalServiceError, LLMOutputError, RegionNotFoundError,
                        VectorStoreUnavailableError)
from src.logging_config import get_logger, setup_logging


def test_setup_logging_is_idempotent_and_uses_env_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    setup_logging(force=True)
    setup_logging()
    root = logging.getLogger("castone")
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1
    assert get_logger("src.agent.workflow").name == "castone.src.agent.workflow"


def test_errors_carry_code_and_user_message():
    e = RegionNotFoundError("중구")
    assert isinstance(e, AppError) and e.code == "REGION_NOT_FOUND"
    assert "중구" in e.user_message and "시/도" in e.user_message
    assert LLMOutputError("briefing", ValueError("bad json")).code == "LLM_OUTPUT_INVALID"
    assert ExternalServiceError("Google Maps", "timeout").code == "EXTERNAL_SERVICE_FAILED"
    assert VectorStoreUnavailableError().code == "VECTOR_STORE_UNAVAILABLE"
