"""장소의 시간대 적합도: 리뷰가 어느 시간대(밤·낮 …)를 말하는지 대표 문장 임베딩으로 추정.

장소마다 규칙을 코드로 넣는 대신, 시간대별 대표 문장(데이터)과 리뷰 벡터의 유사도로 일반화한다.
새 시간대·상황(노을, 일출 등)은 TIME_PROTOTYPES에 문장을 추가하면 된다.
검증(관광지 461곳, 실버 정답 = 리뷰 야경·밤 언급 30%↑): 밤 판별 AUC 0.94, F1 0.71 (MARGIN 0.05, 기준 비율 0.10).
"""
import numpy as np

from src.domain.models import Place
from src.domain.schedule import NIGHT_FIT  # noqa: F401  (검증된 기준값 — 판정은 schedule에서)
from src.retrieval.catalog import get_catalog

TIME_PROTOTYPES = {
    "밤": ["밤에 가면 야경이 정말 예뻐요", "야간 조명이 켜지면 분위기가 좋아요", "밤바다와 불빛이 아름다워요"],
    "낮": ["낮에 산책하기 좋아요", "햇살 좋은 날 둘러보기 좋아요", "오전에 가면 한적해요"],
}
MARGIN = 0.05          # 리뷰가 한 시간대 쪽이라고 볼 코사인 유사도 차이
MAX_REVIEWS = 30


def _unit(m: np.ndarray) -> np.ndarray:
    """행 정규화 (영벡터는 0으로 둔다)."""
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, n, out=np.zeros_like(m), where=n > 0)


def _prototypes(db) -> tuple[list[str], np.ndarray, list[int]]:
    """대표 문장 임베딩 (인덱스당 1회, db에 캐시) → (시간대 목록, 정규화 행렬, 행별 시간대 번호)."""
    cached = getattr(db, "_castone_time_protos", None)
    if cached is None:
        buckets = list(TIME_PROTOTYPES)
        texts = [t for b in buckets for t in TIME_PROTOTYPES[b]]
        owner = [i for i, b in enumerate(buckets) for _ in TIME_PROTOTYPES[b]]
        cached = (buckets, _unit(np.asarray(db.embedding_function.embed_documents(texts), dtype="float32")), owner)
        db._castone_time_protos = cached
    return cached


def attach_time_fit(db, places: list[Place]) -> list[Place]:
    """장소마다 time_fit(시간대별로 기운 리뷰 비율)을 채운 복사본. 카탈로그에 없는 장소는 빈 dict."""
    cat = get_catalog(db)
    buckets, P, owner = _prototypes(db)
    out = []
    for p in places:
        st = cat.places.get((p.name, p.region))
        if st is None or not st.rows:
            out.append(p)
            continue
        V = _unit(np.vstack([db.index.reconstruct(int(i)) for i in st.rows[:MAX_REVIEWS]]).astype("float32"))
        S = V @ P.T                                                      # 리뷰 × 대표 문장
        per_bucket = np.stack([S[:, [j for j, o in enumerate(owner) if o == b]].max(1) for b in range(len(buckets))], 1)
        top2 = np.sort(per_bucket, axis=1)[:, -2:] if len(buckets) > 1 else per_bucket
        lean = per_bucket.argmax(1)
        decided = (top2[:, -1] - top2[:, 0] > MARGIN) & (np.abs(per_bucket).sum(1) > 0)
        fit = {b: float(((lean == k) & decided).mean()) for k, b in enumerate(buckets)}
        out.append(p.model_copy(update={"time_fit": fit}))
    return out
