"""앱 설정: 환경변수 로드, LLM·Google Maps 클라이언트, 인덱스 경로."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from src.llm_factory import make_llm
from src.services.maps_client import RoutesCompatClient
from src.logging_config import get_logger

log = get_logger(__name__)

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
# 문서 인덱스는 쿼리(HuggingFaceEmbeddings = Mean 풀링)와 같은 풀링이어야 함. 원본 CLS 인덱스는 review_faiss
review_faiss = os.path.join(os.path.dirname(current_dir), os.getenv("INDEX_DIR", "review_faiss_mean"))



@lru_cache(maxsize=1)
def get_llm():
    """앱 기본 LLM. import 시점이 아니라 처음 쓸 때 만든다 (LLM_MODEL 없이도 모듈 import 가능)."""
    return make_llm()


GMAPS_API_KEY = os.getenv("GMAPS_API_KEY")
GMAPS_CLIENT = None
if GMAPS_API_KEY:
    GMAPS_CLIENT = RoutesCompatClient(key=GMAPS_API_KEY)
else:
    log.warning("경고: .env 파일에 GMAPS_API_KEY가 설정되지 않았습니다.")
