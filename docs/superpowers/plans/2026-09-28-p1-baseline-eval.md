# P1 베이스라인 평가 체계 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 castone_3의 검색·일정 생성 로직을 **수정하지 않은 채** 실행해, 검색 컴포넌트(A층)와 E2E 일정(B층) 지표 + LLM-as-judge 취향 적합도를 베이스라인 수치로 남긴다.

**Architecture:** 평가 코드는 `eval/`에 두고 기존 `src/`는 인프라 두 곳만 교체한다(LLM 생성 → `init_chat_model`, Google Maps legacy API → Routes API 호환 어댑터). 정답이 정해진 지표(지역·카테고리·슬롯·중복)는 메타데이터로 자동 채점하고, 취향 적합도만 LLM-as-judge로 채점하며 judge는 수기 라벨과의 일치도(가중 kappa)로 검증한다. P1 종료 시점 커밋에 `baseline-v0` 태그를 달아 이후 단계의 비교 기준으로 고정한다.

**Tech Stack:** Python 3.11, LangChain v1(`init_chat_model`, `with_structured_output`, callbacks, `SQLiteCache`), LangGraph 1.0(기존 그래프), FAISS, sentence-transformers(bge-m3-korean), Google Geocoding API + Routes API, Ollama(qwen2.5 7B) / OpenAI(gpt-5-mini), pytest

**Spec:** `docs/superpowers/specs/2026-09-28-castone3-refactor-design.md`

## Global Constraints

- 가상환경: `castone_3/.venv` (Python 3.11). 모든 명령은 `castone_3/`에서 `.venv/bin/python` / `.venv/bin/pytest`로 실행.
- LLM은 `.env`의 `LLM_MODEL`(`provider:model`)로만 결정. 코드에 provider 하드코딩 금지.
- `openai:gpt-5*` 모델에는 `temperature`를 넘기지 않는다(미지원). 그 외 모델은 `temperature=0`.
- 임베딩: `upskyy/bge-m3-korean`, 정규화 없음. 베이스라인 인덱스는 `review_faiss/`(CLS 풀링, 원본 재현).
- **기존 `src/`의 검색·일정 로직은 한 줄도 바꾸지 않는다.** 허용되는 변경은 Task 5의 인프라 교체(LLM 생성, Maps 클라이언트)뿐.
- 공식 before/after 수치는 **같은 모델**로 측정한다. 개발 중에는 Ollama, 공식 측정은 `openai:gpt-5-mini` + `--no-cache`.
- Google Maps: Geocoding API + Routes API만 사용(Directions·Distance Matrix는 legacy라 사용 불가). 한국은 TRANSIT만 결과가 나온다.
- 평가 지역 비교는 반드시 `eval.regions.canonical_region`을 거친다(`서울 강동구` == `서울특별시 강동구`).
- 커밋은 사용자 승인 후 진행한다.

## Review Focus

1. **옛 그래프가 무한 루프로 `GraphRecursionError`에 걸리는 여행 요청** → 러너가 죽지 않고 해당 요청을 `ok=false, error="GraphRecursionError"`로 기록해야 한다. (Task 9 테스트)
2. **일정에 `"추천 장소 없음"`이나 메타데이터에 없는 이름이 들어간 경우** → judge 점수는 `None`, `unmatched` 카운트로 따로 집계하고 평균에서 제외. (Task 10 테스트)
3. **`서울`/`서울특별시` 표기 혼재와 광주광역시/경기도 광주시 혼동** → 채점 함수가 앞의 것은 같은 지역, 뒤의 것은 다른 지역으로 판정. (Task 2 테스트)
4. **Routes API가 경로 없음(짧은 거리, 도보, 같은 지점)을 반환하는 경우** → 어댑터가 legacy와 같은 "결과 없음" 형태(`[]`, element status `ZERO_RESULTS`)를 돌려줘서 옛 코드의 직선거리 fallback이 동작하고 예외가 나지 않아야 한다. (Task 5 테스트)
5. **소형 모델(Ollama)의 judge 구조화 출력 파싱 실패** → 1회 재시도 후 `None`으로 기록하고 실행은 계속. (Task 10 테스트)

---

## File Structure

| 파일 | 책임 |
|---|---|
| `requirements.txt` (수정) | Gemini 제거, LangChain v1·OpenAI·Ollama 추가 |
| `requirements-eval.txt` (신규) | 평가 전용 의존성(pytest, faiss-cpu, sentence-transformers) |
| `src/llm_factory.py` (신규) | `make_llm(model_str)` — provider별 kwargs 처리 |
| `src/maps_compat.py` (신규) | `RoutesCompatClient` — legacy `directions`/`distance_matrix` 시그니처를 Routes API로 구현, 디스크 캐시, 호출 카운터 |
| `src/config.py` (수정) | LLM·Maps 생성부만 교체 |
| `eval/regions.py` | 지역명 정규화·일치 판정 |
| `eval/metadata.py` | `metadata_list.pkl` 로드, (장소, 지역) 인덱스, 리뷰 조회 |
| `eval/build_golden.py` | A층 골든셋 생성 → `eval/data/golden_retrieval.jsonl` |
| `eval/data/golden_trips.jsonl` | B층 여행 요청 20건 (수기) |
| `eval/metrics.py` | 검색·E2E 지표, 가중 kappa, 집계 — 순수 함수 |
| `eval/instrument.py` | LLM 호출·토큰 카운터 callback, LLM 캐시 설정 |
| `eval/legacy_adapter.py` | 옛 `src` 호출 래퍼: `legacy_retrieve`, `run_legacy_trip` |
| `eval/run_retrieval.py` | A층 러너 |
| `eval/run_e2e.py` | B층 러너 |
| `eval/judge.py` | 취향 적합도 judge 체인 |
| `eval/run_judge.py` | judge 러너 + 수기 라벨 시트 내보내기 |
| `eval/judge_agreement.py` | 수기 라벨 vs judge 일치도 |
| `eval/report.py` | 결과 JSON → `docs/eval/baseline.md` |
| `tests/eval/*`, `tests/test_maps_compat.py`, `tests/test_llm_factory.py` | 테스트 |

---

### Task 1: 인덱스 무결성 검증 + 의존성 정리

재임베딩(`scripts/reembed.py`) 산출물이 메타데이터와 짝이 맞는지 확인하고, 이후 태스크가 쓸 의존성을 설치한다.

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-eval.txt`, `tests/__init__.py`, `tests/eval/__init__.py`, `eval/__init__.py`, `tests/test_index_integrity.py`, `pytest.ini`

**Interfaces:**
- Consumes: `review_faiss/`(`index.faiss`, `index.pkl`), `data/metadata_list.pkl`
- Produces: 설치된 환경, `pytest.ini`(마커 `slow` 등록)

- [ ] **Step 1: 의존성 파일 수정**

`requirements.txt` 전체를 다음으로 교체:

```text
emoji==2.15.0
fpdf2==2.8.5
googlemaps==4.10.0
langchain>=1.0,<2
langchain-core>=1.1.0,<2
langchain_community==0.4.1
langchain-openai>=1.0,<2
langchain-ollama>=1.0,<2
langchain_huggingface==1.1.0
langgraph>=1.0.4,<2
pandas==2.3.3
python-dotenv==1.2.1
Requests==2.32.5
streamlit==1.50.0
httpx
aiohttp
```

`requirements-eval.txt` 생성:

```text
-r requirements.txt
faiss-cpu>=1.8
sentence-transformers>=3.0
pytest>=8.0
pytest-asyncio>=0.23
```

`pytest.ini` 생성:

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
markers =
    slow: 인덱스·모델 로드가 필요한 느린 테스트
```

- [ ] **Step 2: 설치**

Run: `.venv/bin/pip install -q -r requirements-eval.txt`
Expected: 오류 없이 종료. `.venv/bin/python -c "from langchain.chat_models import init_chat_model; import langchain_ollama, langchain_openai, faiss, streamlit"` 가 출력 없이 성공.

- [ ] **Step 3: 무결성 테스트 작성**

`tests/__init__.py`, `tests/eval/__init__.py`, `eval/__init__.py`는 빈 파일로 생성.

`tests/test_index_integrity.py`:

```python
import pickle
import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def meta():
    return pickle.load(open("data/metadata_list.pkl", "rb"))


@pytest.fixture(scope="module")
def store():
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="upskyy/bge-m3-korean",
                                encode_kwargs={"normalize_embeddings": False})
    return FAISS.load_local("review_faiss", emb, allow_dangerous_deserialization=True)


def test_counts_match(meta, store):
    assert store.index.ntotal == len(meta) == 195787


def test_docstore_rows_match_metadata(meta, store):
    for i in (0, 1, 97000, 195786):
        doc = store.docstore.search(store.index_to_docstore_id[i])
        assert doc.page_content == meta[i]["text_for_embedding"]
        assert doc.metadata["지역"] == meta[i]["지역"]


def test_vectors_are_cls_of_row_text(meta, store):
    """저장 벡터 i == meta[i] 텍스트의 CLS 임베딩 (원본 레시피 재현 확인)."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("upskyy/bge-m3-korean")
    model = AutoModel.from_pretrained("upskyy/bge-m3-korean").eval()
    ids = [5, 123456]
    with torch.no_grad():
        enc = tok([meta[i]["text_for_embedding"] for i in ids], return_tensors="pt",
                  truncation=True, padding=True, max_length=512)
        cls = model(**enc).last_hidden_state[:, 0].numpy()
    for row, i in enumerate(ids):
        stored = store.index.reconstruct(i)
        cos = float(np.dot(stored, cls[row]) / np.linalg.norm(stored) / np.linalg.norm(cls[row]))
        assert cos > 0.999
```

- [ ] **Step 4: 실행**

Run: `.venv/bin/pytest tests/test_index_integrity.py -v`
Expected: 3 passed. (`review_faiss/`가 없으면 재임베딩이 끝나지 않은 것 — `tail data/reembed.log` 확인 후 대기.)

- [ ] **Step 5: 커밋**

```bash
git add .gitignore requirements.txt requirements-eval.txt pytest.ini eval/__init__.py tests/ scripts/reembed.py docs/
git commit -m "chore: 재임베딩 스크립트, 설계 문서, 평가 의존성 및 인덱스 무결성 테스트"
```

---

### Task 2: 지역 정규화와 메타데이터 접근

**Files:**
- Create: `eval/regions.py`, `eval/metadata.py`
- Test: `tests/eval/test_regions.py`, `tests/eval/test_metadata.py`

**Interfaces:**
- Produces:
  - `canonical_region(text: str) -> tuple[str, str | None]` — `"서울 강동구"` → `("서울특별시", "강동구")`, `"광주"` → `("광주광역시", None)`
  - `region_match(candidate: str, gold_sido: str, gold_sigungu: str | None) -> bool`
  - `SIDO_SHORT: dict[str, str]` — 정식명 → 약칭(`"경기도": "경기"`)
  - `load_metadata() -> list[dict]` (lru_cache)
  - `place_counts() -> dict[tuple[str, str | None, str], int]` — (sido, sigungu, 카테고리) → 고유 장소 수
  - `reviews_for(name: str, region: str, limit: int = 5) -> list[str]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/eval/test_regions.py`:

```python
from eval.regions import canonical_region, region_match, SIDO_SHORT


def test_seoul_short_and_full_are_same():
    assert canonical_region("서울 강동구") == ("서울특별시", "강동구")
    assert canonical_region("서울특별시 강동구") == ("서울특별시", "강동구")


def test_gwangju_is_not_gyeonggi_gwangju():
    assert canonical_region("광주") == ("광주광역시", None)
    assert canonical_region("경기 광주") == ("경기도", "광주시")
    assert not region_match("경기도 광주시", "광주광역시", None)
    assert not region_match("광주광역시 동구", "경기도", "광주시")
    assert region_match("광주광역시 동구", "광주광역시", None)


def test_same_sigungu_name_different_sido():
    assert not region_match("대구광역시 중구", "부산광역시", "중구")
    assert region_match("부산광역시 중구", "부산광역시", "중구")


def test_renamed_provinces():
    assert canonical_region("강원도 강릉시") == ("강원특별자치도", "강릉시")
    assert canonical_region("전라북도 전주시") == ("전북특별자치도", "전주시")


def test_sigungu_short_form_matches():
    # 사용자는 "경기 광주"처럼 '시'를 생략한다
    assert canonical_region("경기 광주") == ("경기도", "광주시")
    assert canonical_region("부산 중구") == ("부산광역시", "중구")


def test_unknown_returns_raw():
    assert canonical_region("") == ("", None)
    assert region_match("", "부산광역시", None) is False


def test_short_names_cover_all_sido():
    assert SIDO_SHORT["경기도"] == "경기"
    assert len(SIDO_SHORT) == 17
```

`tests/eval/test_metadata.py`:

```python
import pytest
from eval.metadata import place_counts, reviews_for

pytestmark = pytest.mark.slow


def test_place_counts_canonicalized():
    c = place_counts()
    assert c[("서울특별시", "강동구", "식당")] > 0
    assert ("서울", "강동구", "식당") not in c


def test_reviews_for_known_place():
    revs = reviews_for("이정태본참치", "부산광역시 중구")
    assert revs and all(isinstance(r, str) for r in revs)
    assert reviews_for("없는장소이름", "부산광역시 중구") == []
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/eval/test_regions.py -v`
Expected: FAIL (`ModuleNotFoundError: eval.regions`)

- [ ] **Step 3: 구현**

`eval/regions.py`:

```python
"""평가용 지역명 정규화. src/ 변경과 무관하게 채점 기준을 고정하기 위해 독립 구현."""

SIDO_SHORT = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
    "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종",
    "경기도": "경기", "강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남",
    "전북특별자치도": "전북", "전라남도": "전남", "경상북도": "경북", "경상남도": "경남",
    "제주특별자치도": "제주",
}
_ALIASES = {short: full for full, short in SIDO_SHORT.items()}
_ALIASES.update({full: full for full in SIDO_SHORT})
_ALIASES.update({
    "서울시": "서울특별시", "부산시": "부산광역시", "대구시": "대구광역시", "인천시": "인천광역시",
    "광주시": "광주광역시", "대전시": "대전광역시", "울산시": "울산광역시", "세종시": "세종특별자치시",
    "강원도": "강원특별자치도", "전라북도": "전북특별자치도", "제주도": "제주특별자치도",
    "충청북": "충청북도", "충청남": "충청남도", "경상북": "경상북도", "경상남": "경상남도",
})
_SUFFIXES = ("시", "군", "구")


def _norm_sigungu(token: str, sido: str) -> str:
    if token.endswith(_SUFFIXES):
        return token
    # "경기 광주" → 광주시. 광역시 하위는 구, 도 하위는 시로 가정 (평가 골든셋 표기 규칙)
    return token + ("구" if sido.endswith(("특별시", "광역시")) else "시")


def canonical_region(text: str) -> tuple[str, str | None]:
    parts = (text or "").split()
    if not parts:
        return ("", None)
    sido = _ALIASES.get(parts[0], parts[0])
    if len(parts) == 1:
        return (sido, None)
    return (sido, _norm_sigungu(parts[1], sido))


def region_match(candidate: str, gold_sido: str, gold_sigungu: str | None) -> bool:
    sido, sigungu = canonical_region(candidate)
    if not sido or sido != gold_sido:
        return False
    return gold_sigungu is None or sigungu == gold_sigungu
```

`eval/metadata.py`:

```python
import pickle
from collections import Counter, defaultdict
from functools import lru_cache

from eval.regions import canonical_region

META_PATH = "data/metadata_list.pkl"


@lru_cache(maxsize=1)
def load_metadata() -> list[dict]:
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def place_counts() -> dict[tuple[str, str | None, str], int]:
    seen = set()
    counts: Counter = Counter()
    for r in load_metadata():
        sido, sigungu = canonical_region(r["지역"])
        key = (sido, sigungu, r["카테고리"], r["장소명"])
        if key not in seen:
            seen.add(key)
            counts[(sido, sigungu, r["카테고리"])] += 1
    return dict(counts)


@lru_cache(maxsize=1)
def _reviews_index() -> dict[tuple[str, str, str | None], list[str]]:
    idx: dict = defaultdict(list)
    for r in load_metadata():
        idx[(r["장소명"], *canonical_region(r["지역"]))].append(r["리뷰"])
    return idx


def reviews_for(name: str, region: str, limit: int = 5) -> list[str]:
    return _reviews_index().get((name, *canonical_region(region)), [])[:limit]
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/pytest tests/eval/test_regions.py tests/eval/test_metadata.py -v`
Expected: 9 passed

- [ ] **Step 5: 커밋**

```bash
git add eval/regions.py eval/metadata.py tests/eval/test_regions.py tests/eval/test_metadata.py
git commit -m "feat(eval): 지역명 정규화와 메타데이터 조회"
```

---

### Task 3: A층 골든셋 생성

**Files:**
- Create: `eval/build_golden.py`, `eval/data/golden_retrieval.jsonl`(생성물)
- Test: `tests/eval/test_build_golden.py`

**Interfaces:**
- Consumes: `canonical_region`, `SIDO_SHORT`, `place_counts`
- Produces: `build_items(seed: int = 42, n_regular: int = 30) -> list[dict]`. 각 item:
  `{"id": str, "group": "ambiguous"|"gwangju"|"regular"|"landmark", "destination": str, "query": str, "category_filter": "식당"|"카페"|"관광지", "gold_sido": str, "gold_sigungu": str | None, "gold_category": str}`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/eval/test_build_golden.py`:

```python
import pytest
from eval.build_golden import build_items

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def items():
    return build_items()


def test_groups_present(items):
    groups = {i["group"] for i in items}
    assert groups == {"ambiguous", "gwangju", "regular", "landmark"}


def test_gwangju_pair(items):
    g = [i for i in items if i["group"] == "gwangju"]
    assert {(i["destination"], i["gold_sido"], i["gold_sigungu"]) for i in g} == {
        ("광주", "광주광역시", None), ("경기 광주", "경기도", "광주시")}
    assert len(g) == 6


def test_ambiguous_covers_same_named_districts(items):
    amb = {(i["gold_sido"], i["gold_sigungu"]) for i in items if i["group"] == "ambiguous"}
    assert ("부산광역시", "중구") in amb and ("대구광역시", "중구") in amb
    assert ("강원특별자치도", "고성군") in amb and ("경상남도", "고성군") in amb


def test_no_seoul_short_duplicates(items):
    assert all(i["gold_sido"] != "서울" for i in items)


def test_ids_unique_and_deterministic(items):
    assert len({i["id"] for i in items}) == len(items)
    assert [i["id"] for i in build_items()] == [i["id"] for i in items]


def test_every_gold_has_enough_places(items):
    from eval.metadata import place_counts
    c = place_counts()
    for i in items:
        if i["gold_sigungu"] is None:
            continue
        assert c.get((i["gold_sido"], i["gold_sigungu"], i["gold_category"]), 0) >= 5, i["id"]
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/eval/test_build_golden.py -v`
Expected: FAIL (`ModuleNotFoundError: eval.build_golden`)

- [ ] **Step 3: 구현**

`eval/build_golden.py`:

```python
"""A층(검색 컴포넌트) 골든셋 생성. 정답은 메타데이터의 지역·카테고리 → 자동 채점."""
import json
import random
from collections import defaultdict
from pathlib import Path

from eval.metadata import place_counts
from eval.regions import SIDO_SHORT

CATS = ["식당", "카페", "관광지"]
QUERY = {"식당": "맛집", "카페": "카페", "관광지": "관광지"}
MIN_PLACES = 5
OUT = Path("eval/data/golden_retrieval.jsonl")

# 목적지는 광역 단위, 쿼리에 지명이 들어가는 경우 (앱은 목적지 기준으로 지역을 해석하므로 약점 측정용)
LANDMARKS = [
    ("부산", "해운대 맛집", "식당", "부산광역시", "해운대구"),
    ("부산", "광안리 카페", "카페", "부산광역시", "수영구"),
    ("부산", "서면 맛집", "식당", "부산광역시", "부산진구"),
    ("서울", "홍대 카페", "카페", "서울특별시", "마포구"),
    ("서울", "강남역 맛집", "식당", "서울특별시", "강남구"),
    ("전북", "전주 한옥마을 관광지", "관광지", "전북특별자치도", "전주시"),
    ("경북", "경주 황리단길 카페", "카페", "경상북도", "경주시"),
    ("강원", "속초 중앙시장 맛집", "식당", "강원특별자치도", "속초시"),
]


def _item(group, dest, query, cat, sido, sigungu):
    return {"id": f"{group}-{sido}-{sigungu or '전체'}-{cat}-{query}", "group": group,
            "destination": dest, "query": query, "category_filter": cat,
            "gold_sido": sido, "gold_sigungu": sigungu, "gold_category": cat}


def build_items(seed: int = 42, n_regular: int = 30) -> list[dict]:
    counts = place_counts()
    ok = lambda s, g, c: counts.get((s, g, c), 0) >= MIN_PLACES
    regions = sorted({(s, g) for (s, g, _c) in counts if g})
    items: list[dict] = []

    # 1) 이름이 같은 시군구 (중구·동구·고성군 …)
    by_name = defaultdict(list)
    for s, g in regions:
        by_name[g].append(s)
    amb = [(s, g) for g, sidos in sorted(by_name.items()) if len(sidos) > 1 for s in sorted(sidos)]
    for n, (s, g) in enumerate(amb):
        cat = next((c for c in CATS[n % 3:] + CATS[:n % 3] if ok(s, g, c)), None)
        if cat:
            items.append(_item("ambiguous", f"{SIDO_SHORT[s]} {g}", QUERY[cat], cat, s, g))

    # 2) 광주광역시 vs 경기도 광주시
    for cat in CATS:
        items.append(_item("gwangju", "광주", QUERY[cat], cat, "광주광역시", None))
        items.append(_item("gwangju", "경기 광주", QUERY[cat], cat, "경기도", "광주시"))

    # 3) 일반 시군구 무작위
    amb_set = set(amb) | {("경기도", "광주시")}
    pool = [(s, g) for s, g in regions if (s, g) not in amb_set and not g.endswith("구")]
    rng = random.Random(seed)
    for n, (s, g) in enumerate(rng.sample(pool, min(n_regular, len(pool)))):
        cat = CATS[n % 3]
        if ok(s, g, cat):
            items.append(_item("regular", f"{SIDO_SHORT[s]} {g}", QUERY[cat], cat, s, g))

    # 4) 지명 쿼리
    for dest, query, cat, s, g in LANDMARKS:
        if ok(s, g, cat):
            items.append(_item("landmark", dest, query, cat, s, g))
    return items


def main():
    items = build_items()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n")
    by = defaultdict(int)
    for i in items:
        by[i["group"]] += 1
    print(f"{len(items)} items → {OUT}  {dict(by)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인 및 생성**

Run: `.venv/bin/pytest tests/eval/test_build_golden.py -v && .venv/bin/python -m eval.build_golden`
Expected: 6 passed, 이어서 `~70 items → eval/data/golden_retrieval.jsonl  {'ambiguous': ~30, 'gwangju': 6, 'regular': ~28, 'landmark': ~8}`

- [ ] **Step 5: 커밋**

```bash
git add eval/build_golden.py eval/data/golden_retrieval.jsonl tests/eval/test_build_golden.py
git commit -m "feat(eval): A층 검색 골든셋 생성기 (동명 지역·광주·지명 쿼리 포함)"
```

---

### Task 4: B층 여행 요청 골든셋

**Files:**
- Create: `eval/data/golden_trips.jsonl`
- Test: `tests/eval/test_golden_trips.py`

**Interfaces:**
- Produces: 한 줄당 `{"id", "destination", "start_location", "days", "group_type", "style", "detail", "gold_sido", "gold_sigungu"}`. `group_type`·`style` 값은 `app.py`의 선택지와 동일.

- [ ] **Step 1: 테스트 작성**

`tests/eval/test_golden_trips.py`:

```python
import json
from pathlib import Path
from eval.regions import SIDO_SHORT

STYLES = {"맛집 탐방", "힐링/휴양", "액티비티", "문화/역사", "자연 감상"}


def load():
    return [json.loads(l) for l in Path("eval/data/golden_trips.jsonl").read_text().splitlines() if l.strip()]


def test_schema_and_values():
    trips = load()
    assert len(trips) == 20
    assert len({t["id"] for t in trips}) == 20
    for t in trips:
        assert 1 <= t["days"] <= 3
        assert t["style"] in STYLES
        assert t["gold_sido"] in SIDO_SHORT
        assert t["detail"]


def test_ambiguous_destinations_included():
    dests = {t["destination"] for t in load()}
    assert {"광주", "경기 광주"} <= dests
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/eval/test_golden_trips.py -v`
Expected: FAIL (`FileNotFoundError`)

- [ ] **Step 3: 데이터 작성**

`eval/data/golden_trips.jsonl`:

```jsonl
{"id": "t01", "destination": "광주", "start_location": "광주송정역", "days": 2, "group_type": "친구", "style": "맛집 탐방", "detail": "떡갈비랑 오리탕 같은 지역 음식 먹고 싶어요", "gold_sido": "광주광역시", "gold_sigungu": null}
{"id": "t02", "destination": "경기 광주", "start_location": "경기광주역", "days": 1, "group_type": "가족", "style": "자연 감상", "detail": "아이랑 같이 가서 걷기 편한 곳이면 좋겠어요", "gold_sido": "경기도", "gold_sigungu": "광주시"}
{"id": "t03", "destination": "부산", "start_location": "부산역", "days": 3, "group_type": "연인", "style": "힐링/휴양", "detail": "바다 보이는 조용한 카페 좋아해요", "gold_sido": "부산광역시", "gold_sigungu": null}
{"id": "t04", "destination": "부산 중구", "start_location": "부산역", "days": 1, "group_type": "친구", "style": "맛집 탐방", "detail": "시장 음식이랑 노포 좋아해요", "gold_sido": "부산광역시", "gold_sigungu": "중구"}
{"id": "t05", "destination": "대구 중구", "start_location": "동대구역", "days": 1, "group_type": "혼자", "style": "문화/역사", "detail": "근대 골목 같은 역사적인 곳 가보고 싶어요", "gold_sido": "대구광역시", "gold_sigungu": "중구"}
{"id": "t06", "destination": "강릉", "start_location": "강릉역", "days": 2, "group_type": "연인", "style": "힐링/휴양", "detail": "커피 좋아하고 바다 산책 원해요", "gold_sido": "강원특별자치도", "gold_sigungu": "강릉시"}
{"id": "t07", "destination": "경주", "start_location": "신경주역", "days": 2, "group_type": "가족", "style": "문화/역사", "detail": "부모님 모시고 유적지 위주로 천천히", "gold_sido": "경상북도", "gold_sigungu": "경주시"}
{"id": "t08", "destination": "전주", "start_location": "전주역", "days": 1, "group_type": "친구", "style": "맛집 탐방", "detail": "비빔밥 말고 현지인 맛집 가고 싶어요", "gold_sido": "전북특별자치도", "gold_sigungu": "전주시"}
{"id": "t09", "destination": "여수", "start_location": "여수엑스포역", "days": 2, "group_type": "연인", "style": "자연 감상", "detail": "야경 좋은 곳이랑 해산물", "gold_sido": "전라남도", "gold_sigungu": "여수시"}
{"id": "t10", "destination": "속초", "start_location": "속초고속버스터미널", "days": 2, "group_type": "친구", "style": "액티비티", "detail": "활동적인 거 하고 저녁엔 회 먹고 싶어요", "gold_sido": "강원특별자치도", "gold_sigungu": "속초시"}
{"id": "t11", "destination": "제주", "start_location": "제주국제공항", "days": 3, "group_type": "가족", "style": "자연 감상", "detail": "오름이나 숲길, 아이가 먹기 좋은 식당", "gold_sido": "제주특별자치도", "gold_sigungu": null}
{"id": "t12", "destination": "서울 종로구", "start_location": "서울역", "days": 1, "group_type": "혼자", "style": "문화/역사", "detail": "궁궐이랑 전통찻집", "gold_sido": "서울특별시", "gold_sigungu": "종로구"}
{"id": "t13", "destination": "인천 중구", "start_location": "인천역", "days": 1, "group_type": "친구", "style": "맛집 탐방", "detail": "차이나타운 말고도 괜찮은 곳", "gold_sido": "인천광역시", "gold_sigungu": "중구"}
{"id": "t14", "destination": "대전", "start_location": "대전역", "days": 1, "group_type": "연인", "style": "맛집 탐방", "detail": "빵지순례 하고 싶어요", "gold_sido": "대전광역시", "gold_sigungu": null}
{"id": "t15", "destination": "수원", "start_location": "수원역", "days": 1, "group_type": "가족", "style": "문화/역사", "detail": "화성 주변 걷기 좋은 코스", "gold_sido": "경기도", "gold_sigungu": "수원시"}
{"id": "t16", "destination": "가평", "start_location": "가평역", "days": 2, "group_type": "친구", "style": "액티비티", "detail": "물놀이나 레저 하고 바비큐", "gold_sido": "경기도", "gold_sigungu": "가평군"}
{"id": "t17", "destination": "통영", "start_location": "통영종합버스터미널", "days": 2, "group_type": "연인", "style": "힐링/휴양", "detail": "섬 풍경이랑 굴 요리", "gold_sido": "경상남도", "gold_sigungu": "통영시"}
{"id": "t18", "destination": "안동", "start_location": "안동역", "days": 1, "group_type": "가족", "style": "문화/역사", "detail": "하회마을이랑 찜닭", "gold_sido": "경상북도", "gold_sigungu": "안동시"}
{"id": "t19", "destination": "춘천", "start_location": "춘천역", "days": 1, "group_type": "친구", "style": "맛집 탐방", "detail": "닭갈비랑 막국수", "gold_sido": "강원특별자치도", "gold_sigungu": "춘천시"}
{"id": "t20", "destination": "양평", "start_location": "양평역", "days": 1, "group_type": "연인", "style": "힐링/휴양", "detail": "강 보이는 조용한 카페와 산책", "gold_sido": "경기도", "gold_sigungu": "양평군"}
```

`app.py`의 `gathering_type` 선택지와 `group_type` 값이 다르면(예: "연인" 대신 "커플") 파일의 값을 `app.py` 선택지로 치환한다. 확인: `grep -n "gathering_type = st" -A2 app.py`

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/pytest tests/eval/test_golden_trips.py -v`
Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add eval/data/golden_trips.jsonl tests/eval/test_golden_trips.py
git commit -m "feat(eval): B층 여행 요청 골든셋 20건"
```

---

### Task 5: 인프라 교체 — LLM 팩토리와 Routes API 호환 Maps 클라이언트

기존 로직을 건드리지 않고 실행 가능하게 만드는 유일한 `src/` 변경.

**Files:**
- Create: `src/llm_factory.py`, `src/maps_compat.py`
- Modify: `src/config.py` (파일 안에 같은 내용이 두 번 반복돼 있음 — 두 곳 모두 교체)
- Test: `tests/test_llm_factory.py`, `tests/test_maps_compat.py`

**Interfaces:**
- Produces:
  - `make_llm(model: str | None = None)` → `BaseChatModel`. `model` 생략 시 `os.environ["LLM_MODEL"]`.
  - `llm_kwargs(model: str) -> dict`
  - `class RoutesCompatClient(googlemaps.Client)`: `directions(origin, destination, mode="driving", departure_time=None, **_) -> list[dict]`, `distance_matrix(origins, destinations, mode="driving", **_) -> dict`, `geocode`, `reverse_geocode` (legacy 반환 형태 유지). 클래스 속성 `calls: int`, `network_calls: int`. 캐시 디렉터리는 `MAPS_CACHE_DIR`(기본 `.cache/maps`), `MAPS_CACHE=0`이면 캐시 끔.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_llm_factory.py`:

```python
from src.llm_factory import llm_kwargs


def test_gpt5_has_no_temperature():
    assert llm_kwargs("openai:gpt-5-mini") == {}


def test_others_temperature_zero():
    assert llm_kwargs("ollama:qwen2.5:7b-instruct-q4_K_M") == {"temperature": 0}
    assert llm_kwargs("openai:gpt-4.1-mini") == {"temperature": 0}
```

`tests/test_maps_compat.py`:

```python
import pytest
from src import maps_compat
from src.maps_compat import RoutesCompatClient

KEY = "AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAK"

ROUTE = {"routes": [{"legs": [{
    "duration": "1683s", "distanceMeters": 4239,
    "startLocation": {"latLng": {"latitude": 37.57, "longitude": 126.97}},
    "endLocation": {"latLng": {"latitude": 37.55, "longitude": 126.98}},
    "steps": [
        {"travelMode": "WALK"},
        {"travelMode": "TRANSIT", "transitDetails": {"transitLine": {
            "nameShort": "402", "name": "간선402", "vehicle": {"name": {"text": "버스"}}}}},
    ]}]}]}


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPS_CACHE_DIR", str(tmp_path))
    RoutesCompatClient.calls = RoutesCompatClient.network_calls = 0
    return RoutesCompatClient(key=KEY)


def test_directions_legacy_shape(client, monkeypatch):
    monkeypatch.setattr(maps_compat.requests, "post", lambda *a, **k: FakeResp(ROUTE))
    leg = client.directions("경복궁", "N서울타워", mode="transit")[0]["legs"][0]
    assert leg["duration"] == {"value": 1683, "text": "28분"}
    assert leg["distance"]["text"] == "4.2km"
    assert leg["start_location"] == {"lat": 37.57, "lng": 126.97}
    assert leg["steps"][0]["travel_mode"] == "WALKING"
    line = leg["steps"][1]["transit_details"]["line"]
    assert line["short_name"] == "402" and line["vehicle"]["name"] == "버스"


def test_directions_no_route_returns_empty_list(client, monkeypatch):
    monkeypatch.setattr(maps_compat.requests, "post", lambda *a, **k: FakeResp({}))
    assert client.directions("A", "B", mode="walking") == []


def test_distance_matrix_legacy_shape(client, monkeypatch):
    rows = [
        {"originIndex": 0, "destinationIndex": 1, "duration": "600s", "condition": "ROUTE_EXISTS"},
        {"originIndex": 1, "destinationIndex": 0, "condition": "ROUTE_NOT_FOUND"},
        {"originIndex": 0, "destinationIndex": 0, "condition": "ROUTE_NOT_FOUND"},
        {"originIndex": 1, "destinationIndex": 1, "condition": "ROUTE_NOT_FOUND"},
    ]
    monkeypatch.setattr(maps_compat.requests, "post", lambda *a, **k: FakeResp(rows))
    m = client.distance_matrix(["A", "B"], ["A", "B"], mode="transit")
    assert m["rows"][0]["elements"][1]["duration"]["value"] == 600
    assert m["rows"][1]["elements"][0] == {"status": "ZERO_RESULTS"}


def test_cache_avoids_second_network_call(client, monkeypatch):
    hits = []
    monkeypatch.setattr(maps_compat.requests, "post", lambda *a, **k: hits.append(1) or FakeResp(ROUTE))
    client.directions("A", "B", mode="transit")
    client.directions("A", "B", mode="transit")
    assert len(hits) == 1
    assert RoutesCompatClient.calls == 2 and RoutesCompatClient.network_calls == 1


def test_http_error_returns_empty_not_raise(client, monkeypatch):
    monkeypatch.setattr(maps_compat.requests, "post", lambda *a, **k: FakeResp({"error": {}}, 403))
    assert client.directions("A", "B", mode="transit") == []
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/test_llm_factory.py tests/test_maps_compat.py -v`
Expected: FAIL (`ModuleNotFoundError: src.llm_factory`)

- [ ] **Step 3: 구현**

`src/llm_factory.py`:

```python
import os

from langchain.chat_models import init_chat_model


def llm_kwargs(model: str) -> dict:
    # gpt-5 계열은 temperature를 지원하지 않음
    return {} if model.startswith("openai:gpt-5") else {"temperature": 0}


def make_llm(model: str | None = None):
    model = model or os.environ["LLM_MODEL"]
    return init_chat_model(model, **llm_kwargs(model))
```

`src/maps_compat.py`:

```python
"""legacy Directions / Distance Matrix 시그니처를 Routes API로 구현한 googlemaps.Client.

신규 GCP 프로젝트는 legacy API를 켤 수 없어서 기존 코드를 그대로 돌리기 위한 어댑터.
반환 형태는 기존 코드(src/tools.py, src/scheduler)가 읽는 필드만 legacy와 동일하게 맞춘다.
"""
import hashlib
import json
import os
from pathlib import Path

import googlemaps
import requests

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
MODES = {"transit": "TRANSIT", "walking": "WALK", "driving": "DRIVE", "bicycling": "BICYCLE"}
ROUTE_MASK = ("routes.legs.duration,routes.legs.distanceMeters,routes.legs.startLocation,"
              "routes.legs.endLocation,routes.legs.steps.travelMode,routes.legs.steps.transitDetails")


def _secs(d: str | None) -> int | None:
    return int(d.rstrip("s")) if d else None


def _min_text(s: int) -> str:
    return f"{s // 3600}시간 {(s % 3600) // 60}분" if s >= 3600 else f"{round(s / 60)}분"


def _latlng(loc: dict) -> dict:
    ll = (loc or {}).get("latLng", {})
    return {"lat": ll.get("latitude"), "lng": ll.get("longitude")}


class RoutesCompatClient(googlemaps.Client):
    calls = 0
    network_calls = 0

    def __init__(self, key: str, **kw):
        super().__init__(key=key, **kw)
        self._api_key = key

    # --- cache ---------------------------------------------------------
    def _cached(self, name: str, args: dict, fetch):
        RoutesCompatClient.calls += 1
        use_cache = os.getenv("MAPS_CACHE", "1") != "0"
        path = None
        if use_cache:
            h = hashlib.sha1(json.dumps([name, args], ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
            path = Path(os.getenv("MAPS_CACHE_DIR", ".cache/maps")) / f"{name}-{h}.json"
            if path.exists():
                return json.loads(path.read_text())
        RoutesCompatClient.network_calls += 1
        result = fetch()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False))
        return result

    def _post(self, url: str, body: dict, mask: str):
        headers = {"X-Goog-Api-Key": self._api_key, "X-Goog-FieldMask": mask,
                   "Content-Type": "application/json"}
        try:
            r = requests.post(url, json=body, headers=headers, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # legacy 코드는 예외 대신 '결과 없음'을 처리하도록 짜여 있음
            print(f"Routes API 실패: {e}")
            return None

    # --- geocoding (Geocoding API는 그대로, 캐시만) ----------------------
    def geocode(self, address, **kw):
        return self._cached("geocode", {"address": address, **kw},
                            lambda: super(RoutesCompatClient, self).geocode(address, **kw))

    def reverse_geocode(self, latlng, **kw):
        return self._cached("reverse", {"latlng": list(latlng), **kw},
                            lambda: super(RoutesCompatClient, self).reverse_geocode(latlng, **kw))

    # --- legacy 시그니처 ------------------------------------------------
    def directions(self, origin, destination, mode="driving", departure_time=None, **_):
        # departure_time은 무시: 재현성(캐시 키 고정)과 Routes API 과거 시각 제약 때문
        body = {"origin": {"address": origin}, "destination": {"address": destination},
                "travelMode": MODES.get(mode, "TRANSIT"), "languageCode": "ko"}

        def fetch():
            data = self._post(ROUTES_URL, body, ROUTE_MASK) or {}
            if not data.get("routes"):
                return []
            leg = data["routes"][0]["legs"][0]
            secs = _secs(leg.get("duration")) or 0
            meters = leg.get("distanceMeters", 0)
            steps = []
            for st in leg.get("steps", []):
                if st.get("travelMode") == "TRANSIT":
                    line = st.get("transitDetails", {}).get("transitLine", {})
                    steps.append({"travel_mode": "TRANSIT", "transit_details": {"line": {
                        "short_name": line.get("nameShort"), "name": line.get("name"),
                        "vehicle": {"name": line.get("vehicle", {}).get("name", {}).get("text", "대중교통")}}}})
                else:
                    steps.append({"travel_mode": "WALKING"})
            return [{"legs": [{
                "duration": {"value": secs, "text": _min_text(secs)},
                "distance": {"value": meters, "text": f"{meters / 1000:.1f}km"},
                "start_location": _latlng(leg.get("startLocation")),
                "end_location": _latlng(leg.get("endLocation")),
                "steps": steps}]}]

        return self._cached("directions", body, fetch)

    def distance_matrix(self, origins, destinations, mode="driving", **_):
        body = {"origins": [{"waypoint": {"address": o}} for o in origins],
                "destinations": [{"waypoint": {"address": d}} for d in destinations],
                "travelMode": MODES.get(mode, "TRANSIT")}

        def fetch():
            data = self._post(MATRIX_URL, body, "originIndex,destinationIndex,duration,condition") or []
            rows = [{"elements": [{"status": "ZERO_RESULTS"} for _ in destinations]} for _ in origins]
            for e in data:
                secs = _secs(e.get("duration"))
                if e.get("condition") == "ROUTE_EXISTS" and secs is not None:
                    rows[e["originIndex"]]["elements"][e["destinationIndex"]] = {
                        "status": "OK", "duration": {"value": secs, "text": _min_text(secs)}}
            return {"rows": rows}

        return self._cached("matrix", body, fetch)
```

`src/config.py` 수정 (파일 안 두 벌 모두 동일하게):

```python
# 변경 전
from langchain_google_genai import ChatGoogleGenerativeAI
...
LLM = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.0)
...
    GMAPS_CLIENT = googlemaps.Client(key=GMAPS_API_KEY)

# 변경 후
from src.llm_factory import make_llm
from src.maps_compat import RoutesCompatClient
...
LLM = make_llm()
...
    GMAPS_CLIENT = RoutesCompatClient(key=GMAPS_API_KEY)
```

- [ ] **Step 4: 통과 확인 + 실제 키 스모크**

Run: `.venv/bin/pytest tests/test_llm_factory.py tests/test_maps_compat.py -v`
Expected: 7 passed

Run:
```bash
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from src.config import GMAPS_CLIENT, LLM
print(GMAPS_CLIENT.directions('경복궁','N서울타워',mode='transit')[0]['legs'][0]['duration'])
print(type(LLM).__name__)"
```
Expected: `{'value': ..., 'text': '..분'}` 과 `ChatOllama`

- [ ] **Step 5: 커밋**

```bash
git add src/llm_factory.py src/maps_compat.py src/config.py tests/test_llm_factory.py tests/test_maps_compat.py
git commit -m "chore: LLM을 init_chat_model로, Maps legacy API를 Routes API 호환 클라이언트로 교체 (로직 변경 없음)"
```

---

### Task 6: 지표 함수

**Files:**
- Create: `eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: `region_match`
- Produces:
  - `retrieval_item_metrics(cands: list[dict], item: dict, k: int) -> dict` — cands 원소는 `{"name", "region", "category"}`. 반환 키: `empty, returned, region_acc, region_hit1, category_acc, diversity` (empty면 나머지 0)
  - `category_group(t: str) -> str`
  - `expected_slots(days: int) -> list[list[str]]`
  - `e2e_trip_metrics(itinerary: list[dict], trip: dict) -> dict` — 반환 키: `n_places, slot_fill, category_adherence, overflow, region_acc, dup_rate, total_move_min`
  - `weighted_kappa(a: list[int], b: list[int], labels=(1,2,3,4,5)) -> float` (quadratic)
  - `mean_or_none(xs) -> float | None`, `aggregate(rows: list[dict], keys: list[str], by: str | None = None) -> dict`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/eval/test_metrics.py`:

```python
import pytest
from eval.metrics import (aggregate, category_group, e2e_trip_metrics, expected_slots,
                          mean_or_none, retrieval_item_metrics, weighted_kappa)

ITEM = {"gold_sido": "부산광역시", "gold_sigungu": "중구", "gold_category": "식당"}


def c(name, region, cat="식당"):
    return {"name": name, "region": region, "category": cat}


def test_retrieval_metrics_basic():
    cands = [c("A", "부산광역시 중구"), c("B", "대구광역시 중구"), c("C", "부산광역시 중구", "카페")]
    m = retrieval_item_metrics(cands, ITEM, k=5)
    assert m["returned"] == 3 and m["empty"] == 0
    assert m["region_acc"] == pytest.approx(2 / 3)
    assert m["region_hit1"] == 1
    assert m["category_acc"] == pytest.approx(2 / 3)
    assert m["diversity"] == 1.0


def test_retrieval_metrics_empty():
    m = retrieval_item_metrics([], ITEM, k=5)
    assert m == {"empty": 1, "returned": 0, "region_acc": 0.0, "region_hit1": 0,
                 "category_acc": 0.0, "diversity": 0.0}


def test_retrieval_diversity_counts_same_place_once():
    m = retrieval_item_metrics([c("A", "부산광역시 중구")] * 2, ITEM, k=5)
    assert m["diversity"] == 0.5


def test_expected_slots():
    assert expected_slots(1) == [["식당", "카페", "관광지", "식당"]]
    assert expected_slots(3) == [["식당", "카페", "관광지", "식당"],
                                 ["관광지", "식당", "카페", "관광지", "식당"], ["관광지"]]


def test_category_group():
    assert category_group("맛집") == "식당"
    assert category_group("커피전문점") == "카페"
    assert category_group("관광지") == "관광지"


def p(name, t, day, addr="부산광역시 중구"):
    return {"name": name, "type": t, "day": day, "address": addr}


def test_e2e_metrics():
    trip = {"days": 2, "gold_sido": "부산광역시", "gold_sigungu": "중구"}
    itin = [p("a", "식당", 1), {"type": "move", "duration_min": 10, "day": 1}, p("b", "카페", 1),
            p("c", "식당", 1), p("d", "식당", 1, "대구광역시 중구"),
            p("e", "관광지", 2), p("e", "관광지", 2)]
    m = e2e_trip_metrics(itin, trip)
    assert m["n_places"] == 6
    assert m["slot_fill"] == pytest.approx(5 / 5)          # day1 4/4, day2 1/1
    assert m["category_adherence"] == pytest.approx(4 / 5)  # day1 3번째 식당≠관광지
    assert m["overflow"] == 1                               # day2 1곳 초과
    assert m["region_acc"] == pytest.approx(5 / 6)
    assert m["dup_rate"] == pytest.approx(1 - 5 / 6)
    assert m["total_move_min"] == 10


def test_e2e_metrics_empty_itinerary():
    m = e2e_trip_metrics([], {"days": 1, "gold_sido": "부산광역시", "gold_sigungu": None})
    assert m["n_places"] == 0 and m["slot_fill"] == 0 and m["region_acc"] is None


def test_weighted_kappa():
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert weighted_kappa([1, 1, 5, 5], [5, 5, 1, 1]) < 0


def test_aggregate_ignores_none_and_groups():
    rows = [{"g": "x", "v": 1.0}, {"g": "x", "v": None}, {"g": "y", "v": 0.0}]
    assert mean_or_none([None, None]) is None
    out = aggregate(rows, ["v"], by="g")
    assert out["all"]["v"] == 0.5 and out["x"]["v"] == 1.0 and out["y"]["n"] == 1
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/eval/test_metrics.py -v`
Expected: FAIL (`ModuleNotFoundError: eval.metrics`)

- [ ] **Step 3: 구현**

`eval/metrics.py`:

```python
"""평가 지표. 모두 순수 함수 — LLM·I/O 없음."""
from collections import defaultdict

from eval.regions import region_match

FIRST = ["식당", "카페", "관광지", "식당"]
MIDDLE = ["관광지", "식당", "카페", "관광지", "식당"]
LAST = ["관광지"]


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def retrieval_item_metrics(cands: list[dict], item: dict, k: int) -> dict:
    top = cands[:k]
    if not top:
        return {"empty": 1, "returned": 0, "region_acc": 0.0, "region_hit1": 0,
                "category_acc": 0.0, "diversity": 0.0}
    reg = [region_match(c["region"], item["gold_sido"], item["gold_sigungu"]) for c in top]
    cat = [c["category"] == item["gold_category"] for c in top]
    return {"empty": 0, "returned": len(top), "region_acc": sum(reg) / len(top),
            "region_hit1": int(reg[0]), "category_acc": sum(cat) / len(top),
            "diversity": len({(c["name"], c["region"]) for c in top}) / len(top)}


def category_group(t: str) -> str:
    t = str(t)
    if any(x in t for x in ["식당", "맛집", "음식", "요리", "레스토랑"]):
        return "식당"
    if any(x in t for x in ["카페", "커피", "베이커리", "디저트"]):
        return "카페"
    return "관광지"


def expected_slots(days: int) -> list[list[str]]:
    # 기존 graph_flow 규칙: 1일차 4곳(점심 시작), 중간 날 5곳, 마지막 날 1곳. 1일 여행은 1일차 규칙.
    return [FIRST if d == 1 else LAST if d == days else MIDDLE for d in range(1, days + 1)]


def e2e_trip_metrics(itinerary: list[dict], trip: dict) -> dict:
    places = [x for x in itinerary if x.get("type") != "move"]
    moves = [x for x in itinerary if x.get("type") == "move"]
    exp = expected_slots(trip["days"])
    by_day = defaultdict(list)
    for x in places:
        by_day[int(x.get("day", 1))].append(category_group(x.get("type", "")))
    total = sum(len(e) for e in exp)
    filled = adhered = overflow = 0
    for d, e in enumerate(exp, start=1):
        actual = by_day.get(d, [])
        filled += min(len(actual), len(e))
        adhered += sum(a == b for a, b in zip(actual, e))
        overflow += max(0, len(actual) - len(e))
    overflow += sum(len(v) for d, v in by_day.items() if d > len(exp))
    names = [x.get("name") for x in places]
    return {
        "n_places": len(places),
        "slot_fill": filled / total,
        "category_adherence": adhered / total,
        "overflow": overflow,
        "region_acc": mean_or_none([float(region_match(x.get("address", ""), trip["gold_sido"],
                                                       trip["gold_sigungu"])) for x in places]),
        "dup_rate": (1 - len(set(names)) / len(names)) if names else 0.0,
        "total_move_min": sum(int(m.get("duration_min", 0)) for m in moves),
    }


def weighted_kappa(a: list[int], b: list[int], labels=(1, 2, 3, 4, 5)) -> float:
    n, k = len(a), len(labels)
    pos = {l: i for i, l in enumerate(labels)}
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[pos[x]][pos[y]] += 1
    ra = [sum(r) for r in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = lambda i, j: (i - j) ** 2 / (k - 1) ** 2
    num = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * ra[i] * cb[j] / n for i in range(k) for j in range(k))
    return 1.0 - num / den if den else 1.0


def aggregate(rows: list[dict], keys: list[str], by: str | None = None) -> dict:
    groups = {"all": rows}
    if by:
        for r in rows:
            groups.setdefault(r[by], []).append(r)
    return {g: {"n": len(rs), **{k: mean_or_none([r.get(k) for r in rs]) for k in keys}}
            for g, rs in groups.items()}
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/pytest tests/eval/test_metrics.py -v`
Expected: 9 passed

- [ ] **Step 5: 커밋**

```bash
git add eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat(eval): 검색·E2E 지표, 가중 kappa, 집계 함수"
```

---

### Task 7: 계측 — LLM 호출·토큰 카운터와 캐시

**Files:**
- Create: `eval/instrument.py`
- Test: `tests/eval/test_instrument.py`

**Interfaces:**
- Produces:
  - `class LLMCallCounter(BaseCallbackHandler)`: 속성 `calls`, `input_tokens`, `output_tokens`, 메서드 `reset()`, `snapshot() -> dict`
  - `attach_counter(llm, counter) -> None` — `llm.callbacks`에 counter 추가 (체인·bind_tools에 이미 묶인 같은 인스턴스에도 적용됨)
  - `enable_llm_cache(path: str = ".cache/llm.sqlite") -> None`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/eval/test_instrument.py`:

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate

from eval.instrument import LLMCallCounter, attach_counter


def test_counts_calls_through_prebuilt_chain():
    llm = FakeListChatModel(responses=["a", "b", "c"])
    chain = ChatPromptTemplate.from_template("{x}") | llm   # 계측 전에 만든 체인
    counter = LLMCallCounter()
    attach_counter(llm, counter)
    chain.invoke({"x": "1"})
    llm.invoke("2")
    assert counter.calls == 2
    snap = counter.snapshot()
    counter.reset()
    assert snap["llm_calls"] == 2 and counter.calls == 0
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/eval/test_instrument.py -v`
Expected: FAIL (`ModuleNotFoundError: eval.instrument`)

- [ ] **Step 3: 구현**

`eval/instrument.py`:

```python
from langchain_core.callbacks import BaseCallbackHandler


class LLMCallCounter(BaseCallbackHandler):
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = self.input_tokens = self.output_tokens = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.calls += 1

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.calls += 1

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)

    def snapshot(self) -> dict:
        return {"llm_calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens}


def attach_counter(llm, counter: LLMCallCounter) -> None:
    llm.callbacks = [*(llm.callbacks or []), counter]


def enable_llm_cache(path: str = ".cache/llm.sqlite") -> None:
    from pathlib import Path
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=path))
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/pytest tests/eval/test_instrument.py -v`
Expected: 1 passed

- [ ] **Step 5: 커밋**

```bash
git add eval/instrument.py tests/eval/test_instrument.py
git commit -m "feat(eval): LLM 호출·토큰 카운터와 SQLite LLM 캐시"
```

---

### Task 8: A층 러너 — 옛 검색 파이프라인 재현

`find_and_select_best_place`(`src/tools.py`)의 검색 부분을 **호출 순서 그대로** 재현하되, 최종 1곳이 아니라 후보 목록을 돌려받아 @k로 채점한다. 앱은 `execute_tools`에서 `anchor = current_anchor or destination`을 주입하므로 여기서도 `anchor=destination`.

**Files:**
- Create: `eval/legacy_adapter.py`, `eval/run_retrieval.py`
- Test: `tests/eval/test_legacy_adapter.py`

**Interfaces:**
- Consumes: `src.tools`의 `region_chain`, `query_gen_chain`, `resolve_admin_region`, `get_coordinates`, `calculate_haversine_distance`, `_search_docs`, `_filter_candidates`; `retrieval_item_metrics`, `aggregate`, `LLMCallCounter`, `attach_counter`, `enable_llm_cache`
- Produces:
  - `doc_to_cand(doc) -> dict` (`{"name", "region", "category"}`)
  - `async legacy_retrieve(query, destination, category_filter, user_info="", anchor=None) -> dict` — 키 `target_region, queries, candidates(list[Document]), fallback_used`
  - CLI `python -m eval.run_retrieval --k 5 --out eval/results/retrieval_baseline.json [--no-cache] [--limit N]`

- [ ] **Step 1: 실패하는 테스트 작성** (tools 내부 함수를 가짜로 바꿔 흐름만 검증)

`tests/eval/test_legacy_adapter.py`:

```python
import types
import pytest
from langchain_core.documents import Document

from eval import legacy_adapter


def doc(name, region, cat="식당"):
    return Document(page_content="x", metadata={"장소명": name, "지역": region, "카테고리": cat})


class Chain:
    def __init__(self, out):
        self.out = out

    async def ainvoke(self, _):
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


@pytest.fixture
def fake_tools(monkeypatch):
    calls = {"search": []}

    async def search(q, k=20):
        calls["search"].append((q, k))
        return [doc("A", "부산광역시 중구"), doc("A", "부산광역시 중구"), doc("B", "부산광역시 중구")]

    async def filt(docs, target_region, exclude, category_filter):
        return list(docs)

    async def coords(_):
        return (35.1, 129.0)

    async def resolve(_a, dest):
        return "부산광역시"

    T = types.SimpleNamespace(
        region_chain=Chain("부산광역시 중구"), query_gen_chain=Chain("부산 중구 맛집, 부산 중구 노포"),
        resolve_admin_region=resolve, get_coordinates=coords,
        calculate_haversine_distance=lambda *a: 1.0, _search_docs=search, _filter_candidates=filt)
    monkeypatch.setattr(legacy_adapter, "_tools", lambda: T)
    return calls


async def test_multiquery_search_and_name_dedupe(fake_tools):
    r = await legacy_adapter.legacy_retrieve("맛집", "부산 중구", "식당")
    assert r["target_region"] == "부산광역시 중구"
    assert r["queries"] == ["부산 중구 맛집", "부산 중구 노포"]
    assert fake_tools["search"] == [("부산 중구 맛집", 50), ("부산 중구 노포", 50)]
    assert [d.metadata["장소명"] for d in r["candidates"]] == ["A", "B"]
    assert r["fallback_used"] is False


async def test_region_chain_failure_falls_back_to_maps(fake_tools, monkeypatch):
    T = legacy_adapter._tools()
    T.region_chain = Chain(RuntimeError("llm down"))
    r = await legacy_adapter.legacy_retrieve("맛집", "부산", "식당")
    assert r["target_region"] == "부산광역시"


def test_doc_to_cand():
    assert legacy_adapter.doc_to_cand(doc("A", "부산광역시 중구")) == {
        "name": "A", "region": "부산광역시 중구", "category": "식당"}
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/eval/test_legacy_adapter.py -v`
Expected: FAIL (`ModuleNotFoundError: eval.legacy_adapter`)

- [ ] **Step 3: 구현**

`eval/legacy_adapter.py`:

```python
"""기존 src/ 코드를 수정 없이 호출하는 어댑터. baseline-v0 태그 시점의 src에서만 유효."""
import asyncio


def _tools():
    from src import tools
    return tools


def doc_to_cand(doc) -> dict:
    md = doc.metadata
    return {"name": md.get("장소명", ""), "region": md.get("지역", ""), "category": md.get("카테고리", "")}


async def legacy_retrieve(query, destination, category_filter, user_info="", anchor=None) -> dict:
    """src/tools.py find_and_select_best_place의 검색 단계 재현 (추천사 생성·최종 1곳 선택 제외)."""
    T = _tools()
    anchor = anchor if anchor is not None else destination
    target_input = anchor if anchor else query
    target_region = ""
    try:
        target_region = (await T.region_chain.ainvoke({"query": target_input, "destination": destination})).strip()
    except Exception as e:
        print(f"region_chain 실패: {e}")
    if not target_region or target_region == destination:
        if anchor:
            target_region = await T.resolve_admin_region(anchor, destination)
        else:
            t_inp = query if destination in query else f"{destination} {query}"
            target_region = await T.resolve_admin_region(t_inp, destination)
    target_region = target_region.strip()

    center_lat = center_lng = None
    center_place = anchor if anchor else target_region
    if center_place:
        center_lat, center_lng = await T.get_coordinates(center_place)

    try:
        out = await T.query_gen_chain.ainvoke({"target_region": target_region, "query": query,
                                                "user_info": user_info, "category_filter": category_filter})
        search_queries = [q.strip() for q in out.split(",") if q.strip()]
    except Exception:
        search_queries = [f"{target_region} {query} {category_filter}"]

    results = await asyncio.gather(*[T._search_docs(q, k=50) for q in search_queries])
    seen, aggregated = set(), []
    for docs in results:
        for d in docs:
            name = d.metadata.get("장소명", "")
            if name and name not in seen:
                seen.add(name)
                aggregated.append(d)
    candidates = await T._filter_candidates(aggregated, target_region, [], category_filter)

    fallback_used = False
    if not candidates:
        fallback_used = True
        docs_v2 = await T._search_docs(f"{target_region} {query} {category_filter}", k=30)
        candidates = await T._filter_candidates(docs_v2, target_region, [], category_filter)
        if candidates and center_lat and center_lng:
            scored = []
            for d in candidates[:5]:
                md = d.metadata
                addr = md.get("지역", "") or md.get("road_address", "") or md.get("상세 주소", "")
                lat, lng = await T.get_coordinates(addr)
                dist = T.calculate_haversine_distance(center_lat, center_lng, lat, lng) if lat and lng else 9999.0
                scored.append((dist, d))
            scored.sort(key=lambda x: x[0])
            candidates = [d for _, d in scored]
    return {"target_region": target_region, "queries": search_queries,
            "candidates": candidates, "fallback_used": fallback_used}
```

`eval/run_retrieval.py`:

```python
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from eval.instrument import LLMCallCounter, attach_counter, enable_llm_cache
from eval.metrics import aggregate, retrieval_item_metrics

KEYS = ["empty", "returned", "region_acc", "region_hit1", "category_acc", "diversity", "llm_calls", "latency_s"]


async def main(args):
    load_dotenv(".env")
    if not args.no_cache:
        enable_llm_cache()
    from src.config import LLM
    from eval.legacy_adapter import doc_to_cand, legacy_retrieve
    counter = LLMCallCounter()
    attach_counter(LLM, counter)

    items = [json.loads(l) for l in Path(args.golden).read_text().splitlines() if l.strip()][: args.limit]
    rows = []
    for n, it in enumerate(items, 1):
        counter.reset()
        t0 = time.time()
        try:
            r = await legacy_retrieve(it["query"], it["destination"], it["category_filter"])
            cands = [doc_to_cand(d) for d in r["candidates"]]
            row = {**it, "ok": True, "target_region": r["target_region"], "queries": r["queries"],
                   "fallback_used": r["fallback_used"], "top": cands[: args.k],
                   **retrieval_item_metrics(cands, it, args.k)}
        except Exception as e:
            row = {**it, "ok": False, "error": repr(e), **retrieval_item_metrics([], it, args.k)}
        row.update(latency_s=time.time() - t0, **counter.snapshot())
        rows.append(row)
        print(f"[{n}/{len(items)}] {it['id']}  region_acc={row['region_acc']:.2f}  cat={row['category_acc']:.2f}")

    out = {"llm_model": os.environ["LLM_MODEL"], "k": args.k, "cache": not args.no_cache,
           "summary": aggregate(rows, KEYS, by="group"), "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/data/golden_retrieval.jsonl")
    ap.add_argument("--out", default="eval/results/retrieval_baseline.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    asyncio.run(main(ap.parse_args()))
```

- [ ] **Step 4: 단위 테스트 통과 + 스모크 실행**

Run: `.venv/bin/pytest tests/eval/test_legacy_adapter.py -v`
Expected: 3 passed

Ollama 서버 확인 후 3건만 실행:
```bash
ollama serve >/dev/null 2>&1 &   # 이미 실행 중이면 생략
.venv/bin/python -m eval.run_retrieval --limit 3 --out eval/results/_smoke_retrieval.json
```
Expected: `[1/3] ... region_acc=... cat=...` 3줄과 summary JSON. `eval/results/_smoke_retrieval.json` 생성. (스모크 파일은 커밋하지 않음)

- [ ] **Step 5: 커밋**

```bash
git add eval/legacy_adapter.py eval/run_retrieval.py tests/eval/test_legacy_adapter.py
git commit -m "feat(eval): 기존 검색 파이프라인 재현 어댑터와 A층 러너"
```

---

### Task 9: B층 러너 — 옛 그래프로 여행 일정 생성

`pages/trip_planner.py`의 초기 상태·프롬프트와 `app.py`의 선호 문자열 조립을 그대로 재현해 `build_graph()`를 호출한다.

**Files:**
- Modify: `eval/legacy_adapter.py` (함수 추가)
- Create: `eval/run_e2e.py`
- Test: `tests/eval/test_legacy_trip.py`

**Interfaces:**
- Consumes: `src.graph_flow.build_graph`, `e2e_trip_metrics`, `aggregate`, 계측 함수, `RoutesCompatClient.calls/network_calls`
- Produces:
  - `build_preference(trip: dict) -> str`
  - `initial_state(trip: dict, start: date) -> dict`
  - `async run_legacy_trip(app, trip: dict, start: date) -> dict` — 키 `ok, error, itinerary, final_message`
  - CLI `python -m eval.run_e2e --out eval/results/e2e_baseline.json [--no-cache] [--limit N]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/eval/test_legacy_trip.py`:

```python
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from eval.legacy_adapter import build_preference, initial_state, run_legacy_trip

TRIP = {"id": "t", "destination": "부산", "start_location": "부산역", "days": 2,
        "group_type": "친구", "style": "맛집 탐방", "detail": "노포 좋아요"}


def test_preference_matches_app_format():
    assert build_preference(TRIP) == "- 동행: 친구\n- 스타일: 맛집 탐방\n- 출발지: 부산역\n- 상세 요청: 노포 좋아요"


def test_initial_state_mirrors_trip_planner():
    s = initial_state(TRIP, date(2026, 10, 1))
    assert s["dates"] == "2026년 10월 01일부터 2일간"
    assert s["total_days"] == 2 and s["dialog_stage"] == "planning"
    assert s["style"] == s["preference"]          # 원본 앱의 동작 그대로
    assert isinstance(s["messages"][0], HumanMessage)
    assert "목적지: 부산" in s["messages"][0].content


class App:
    def __init__(self, result=None, exc=None):
        self.result, self.exc = result, exc

    async def ainvoke(self, state, config=None):
        if self.exc:
            raise self.exc
        return self.result


async def test_run_records_recursion_error_instead_of_crashing():
    r = await run_legacy_trip(App(exc=GraphRecursionError("limit")), TRIP, date(2026, 10, 1))
    assert r["ok"] is False and r["error"] == "GraphRecursionError" and r["itinerary"] == []


async def test_run_returns_itinerary_and_last_ai_message():
    res = {"itinerary": [{"name": "a", "type": "식당", "day": 1}],
           "messages": [HumanMessage("q"), AIMessage("계획표")]}
    r = await run_legacy_trip(App(result=res), TRIP, date(2026, 10, 1))
    assert r["ok"] and r["itinerary"][0]["name"] == "a" and r["final_message"] == "계획표"
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/eval/test_legacy_trip.py -v`
Expected: FAIL (`ImportError: cannot import name 'build_preference'`)

- [ ] **Step 3: 구현**

`eval/legacy_adapter.py` 하단에 추가:

```python
from datetime import date


def build_preference(trip: dict) -> str:
    """app.py의 pref_list 조립 재현."""
    lines = [f"- 동행: {trip['group_type']}", f"- 스타일: {trip['style']}"]
    if trip.get("start_location"):
        lines.append(f"- 출발지: {trip['start_location']}")
    if trip.get("detail", "").strip():
        lines.append(f"- 상세 요청: {trip['detail']}")
    return "\n".join(lines)


def initial_state(trip: dict, start: date) -> dict:
    """pages/trip_planner.py run_ai_agent()의 current_state + 초기 프롬프트 재현."""
    from langchain_core.messages import HumanMessage
    days = trip["days"]
    dates = f"{start.strftime('%Y년 %m월 %d일')}부터 {days}일간"
    pref = build_preference(trip)
    prompt = f"""
    안녕하세요! 아래 정보로 여행 계획을 세워주세요.
    - 목적지: {trip['destination']}
    - 일정: {dates} (총 {days}일)
    - 스타일: {pref}
    - 동행: {trip['group_type']}
    
    날씨 확인 후, 1일차 일정부터 바로 시작해주세요.
    """
    return {"messages": [HumanMessage(content=prompt)], "itinerary": [],
            "destination": trip["destination"], "dates": dates, "group_type": trip["group_type"],
            "style": pref, "preference": pref, "total_days": days, "current_weather": "",
            "show_pdf_button": False, "current_anchor": trip["destination"],
            "dialog_stage": "planning", "last_deleted_spot": None, "ban_list": []}


async def run_legacy_trip(app, trip: dict, start: date) -> dict:
    from langchain_core.messages import AIMessage
    config = {"configurable": {"thread_id": f"eval-{trip['id']}"}, "recursion_limit": 50}
    try:
        res = await app.ainvoke(initial_state(trip, start), config=config)
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "detail": str(e)[:300],
                "itinerary": [], "final_message": ""}
    ai = [m for m in res.get("messages", []) if isinstance(m, AIMessage) and m.content]
    return {"ok": True, "error": None, "itinerary": res.get("itinerary", []),
            "final_message": ai[-1].content if ai else ""}
```

`eval/run_e2e.py`:

```python
import argparse
import asyncio
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from eval.instrument import LLMCallCounter, attach_counter, enable_llm_cache
from eval.metrics import aggregate, e2e_trip_metrics

KEYS = ["ok", "n_places", "slot_fill", "category_adherence", "overflow", "region_acc", "dup_rate",
        "total_move_min", "llm_calls", "input_tokens", "output_tokens", "maps_calls", "latency_s"]


async def main(args):
    load_dotenv(".env")
    if not args.no_cache:
        enable_llm_cache()
    from src.config import GMAPS_CLIENT, LLM
    from src.graph_flow import build_graph
    from src.maps_compat import RoutesCompatClient
    from eval.legacy_adapter import run_legacy_trip

    counter = LLMCallCounter()
    attach_counter(LLM, counter)
    app = build_graph()
    start = date.today() + timedelta(days=1)
    trips = [json.loads(l) for l in Path(args.golden).read_text().splitlines() if l.strip()][: args.limit]
    rows = []
    for n, trip in enumerate(trips, 1):
        counter.reset()
        m0, net0 = RoutesCompatClient.calls, RoutesCompatClient.network_calls
        t0 = time.time()
        r = await run_legacy_trip(app, trip, start)
        row = {**trip, **r, "latency_s": time.time() - t0, **counter.snapshot(),
               "maps_calls": RoutesCompatClient.calls - m0,
               "maps_network_calls": RoutesCompatClient.network_calls - net0,
               **e2e_trip_metrics(r["itinerary"], trip)}
        row["ok"] = float(row["ok"])
        rows.append(row)
        print(f"[{n}/{len(trips)}] {trip['id']} ok={r['ok']} places={row['n_places']} "
              f"region={row['region_acc']} llm={row['llm_calls']} {row['latency_s']:.0f}s")

    out = {"llm_model": os.environ["LLM_MODEL"], "start_date": start.isoformat(),
           "cache": not args.no_cache, "summary": aggregate(rows, KEYS), "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/data/golden_trips.jsonl")
    ap.add_argument("--out", default="eval/results/e2e_baseline.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    asyncio.run(main(ap.parse_args()))
```

- [ ] **Step 4: 통과 확인 + 1건 스모크**

Run: `.venv/bin/pytest tests/eval/test_legacy_trip.py -v`
Expected: 4 passed

Run: `.venv/bin/python -m eval.run_e2e --limit 1 --out eval/results/_smoke_e2e.json`
Expected: `[1/1] t01 ok=True places=N ...` (소형 모델에서 `ok=False error=GraphRecursionError`가 나와도 러너는 정상 종료해야 함 — 그 자체가 베이스라인 결과)

- [ ] **Step 5: 커밋**

```bash
git add eval/legacy_adapter.py eval/run_e2e.py tests/eval/test_legacy_trip.py
git commit -m "feat(eval): 기존 LangGraph로 여행 일정을 생성하는 B층 러너"
```

---

### Task 10: LLM-as-judge 취향 적합도 + judge 검증

**Files:**
- Create: `eval/judge.py`, `eval/run_judge.py`, `eval/judge_agreement.py`
- Test: `tests/eval/test_judge.py`

**Interfaces:**
- Consumes: `reviews_for`, `build_preference`, `weighted_kappa`, `mean_or_none`, `make_llm`
- Produces:
  - `class JudgeScore(BaseModel)`: `score: Literal[1,2,3,4,5]`, `rationale: str`
  - `build_judge_chain(llm)` → Runnable(`{"request","place","category","region","reviews"}` → `JudgeScore`)
  - `async judge_place(chain, trip, place) -> dict` — 키 `score: int | None, rationale, status: "ok"|"unmatched"|"parse_error"`
  - CLI `python -m eval.run_judge --e2e eval/results/e2e_baseline.json --out eval/results/judge_baseline.json [--sheet eval/data/judge_labels.csv]`
  - CLI `python -m eval.judge_agreement --judge eval/results/judge_baseline.json --labels eval/data/judge_labels.csv`
  - `.env` 키 `JUDGE_MODEL` (없으면 `LLM_MODEL`)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/eval/test_judge.py`:

```python
import pytest
from langchain_core.runnables import RunnableLambda

from eval import judge
from eval.judge import JudgeScore, judge_place

TRIP = {"destination": "부산", "start_location": "", "days": 1, "group_type": "친구",
        "style": "맛집 탐방", "detail": "노포"}


@pytest.fixture
def reviews(monkeypatch):
    monkeypatch.setattr(judge, "reviews_for", lambda name, region, limit=5: ["오래된 노포, 맛있음"] if name == "A" else [])


async def test_ok(reviews):
    chain = RunnableLambda(lambda x: JudgeScore(score=4, rationale="노포 언급"))
    r = await judge_place(chain, TRIP, {"name": "A", "type": "식당", "address": "부산광역시 중구"})
    assert r == {"score": 4, "rationale": "노포 언급", "status": "ok"}


async def test_unmatched_place_is_not_scored(reviews):
    chain = RunnableLambda(lambda x: pytest.fail("호출되면 안 됨"))
    for name in ["추천 장소 없음", "없는가게"]:
        r = await judge_place(chain, TRIP, {"name": name, "type": "식당", "address": "부산광역시 중구"})
        assert r["score"] is None and r["status"] == "unmatched"


async def test_parse_error_recorded(reviews):
    def boom(_):
        raise ValueError("bad json")
    r = await judge_place(RunnableLambda(boom), TRIP, {"name": "A", "type": "식당", "address": "부산광역시 중구"})
    assert r["score"] is None and r["status"] == "parse_error"
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/eval/test_judge.py -v`
Expected: FAIL (`ModuleNotFoundError: eval.judge`)

- [ ] **Step 3: 구현**

`eval/judge.py`:

```python
"""취향 적합도 LLM-as-judge. 근거는 해당 장소의 실제 리뷰로 한정한다."""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from eval.legacy_adapter import build_preference
from eval.metadata import reviews_for


class JudgeScore(BaseModel):
    score: Literal[1, 2, 3, 4, 5] = Field(description="사용자 요청 적합도 (1=전혀 안 맞음, 5=매우 잘 맞음)")
    rationale: str = Field(description="판단 근거가 된 리뷰 내용을 한두 문장으로")


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 여행 추천 품질 평가자입니다. 아래 '리뷰'만 근거로, 이 장소가 사용자 요청에 얼마나 맞는지 1~5점으로 채점하세요.\n"
     "5: 요청의 핵심 취향이 리뷰로 분명히 뒷받침됨\n4: 대체로 맞고 어긋나는 근거 없음\n"
     "3: 판단할 근거가 부족하거나 반반\n2: 일부 요청과 어긋나는 근거가 있음\n1: 요청과 명백히 반대\n"
     "리뷰에 없는 사실을 추측하지 마세요. 지역·카테고리 일치는 별도로 채점하므로 평가하지 마세요."),
    ("human", "[사용자 요청]\n{request}\n\n[장소] {place} ({category}, {region})\n\n[리뷰]\n{reviews}"),
])


def build_judge_chain(llm):
    return (PROMPT | llm.with_structured_output(JudgeScore)).with_retry(stop_after_attempt=2)


async def judge_place(chain, trip: dict, place: dict) -> dict:
    revs = reviews_for(place.get("name", ""), place.get("address", ""))
    if not revs:
        return {"score": None, "rationale": "", "status": "unmatched"}
    try:
        s = await chain.ainvoke({"request": build_preference(trip), "place": place["name"],
                                 "category": place.get("type", ""), "region": place.get("address", ""),
                                 "reviews": "\n".join(f"- {r[:300]}" for r in revs)})
        return {"score": int(s.score), "rationale": s.rationale, "status": "ok"}
    except Exception as e:
        return {"score": None, "rationale": repr(e)[:200], "status": "parse_error"}
```

`eval/run_judge.py`:

```python
import argparse
import asyncio
import csv
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

from eval.instrument import enable_llm_cache
from eval.judge import build_judge_chain, judge_place
from eval.legacy_adapter import build_preference
from eval.metadata import reviews_for
from eval.metrics import mean_or_none


async def main(args):
    load_dotenv(".env")
    if not args.no_cache:
        enable_llm_cache()
    from src.llm_factory import make_llm
    model = os.getenv("JUDGE_MODEL") or os.environ["LLM_MODEL"]
    chain = build_judge_chain(make_llm(model))
    e2e = json.loads(Path(args.e2e).read_text())
    judged = []
    for trip in e2e["rows"]:
        for place in [x for x in trip["itinerary"] if x.get("type") != "move"]:
            r = await judge_place(chain, trip, place)
            judged.append({"trip_id": trip["id"], "name": place.get("name"),
                           "region": place.get("address", ""), "category": place.get("type", ""), **r})
            print(f"{trip['id']} {place.get('name')}: {r['status']} {r['score']}")

    status = {s: sum(j["status"] == s for j in judged) for s in ("ok", "unmatched", "parse_error")}
    per_trip = {}
    for j in judged:
        per_trip.setdefault(j["trip_id"], []).append(j["score"])
    out = {"judge_model": model, "status_counts": status,
           "mean_score": mean_or_none([j["score"] for j in judged]),
           "mean_trip_score": mean_or_none([mean_or_none(v) for v in per_trip.values()]),
           "rows": judged}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print({k: out[k] for k in ("judge_model", "status_counts", "mean_score", "mean_trip_score")})

    if args.sheet and not Path(args.sheet).exists():
        trips = {t["id"]: t for t in e2e["rows"]}
        scored = [j for j in judged if j["status"] == "ok"]
        sample = random.Random(0).sample(scored, min(40, len(scored)))
        with open(args.sheet, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["idx", "trip_id", "request", "place", "category", "region", "reviews", "human_score"])
            for n, j in enumerate(sample):
                w.writerow([n, j["trip_id"], build_preference(trips[j["trip_id"]]), j["name"], j["category"],
                            j["region"], "\n".join(reviews_for(j["name"], j["region"])), ""])
        print(f"수기 라벨 시트 → {args.sheet} (judge 점수는 숨김, human_score 열을 1~5로 채우세요)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="eval/results/e2e_baseline.json")
    ap.add_argument("--out", default="eval/results/judge_baseline.json")
    ap.add_argument("--sheet", default="eval/data/judge_labels.csv")
    ap.add_argument("--no-cache", action="store_true")
    asyncio.run(main(ap.parse_args()))
```

`eval/judge_agreement.py`:

```python
import argparse
import csv
import json
from pathlib import Path

from eval.metrics import weighted_kappa


def main(args):
    judged = {(j["trip_id"], j["name"]): j["score"]
              for j in json.loads(Path(args.judge).read_text())["rows"] if j["score"] is not None}
    human, model = [], []
    with open(args.labels, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["human_score"].strip():
                key = (row["trip_id"], row["place"])
                if key in judged:
                    human.append(int(row["human_score"]))
                    model.append(judged[key])
    exact = sum(h == m for h, m in zip(human, model)) / len(human)
    within1 = sum(abs(h - m) <= 1 for h, m in zip(human, model)) / len(human)
    print(json.dumps({"n": len(human), "quadratic_weighted_kappa": round(weighted_kappa(human, model), 3),
                      "exact_agreement": round(exact, 3), "within_1": round(within1, 3)}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="eval/results/judge_baseline.json")
    ap.add_argument("--labels", default="eval/data/judge_labels.csv")
    main(ap.parse_args())
```

`.env`에 한 줄 추가 (값은 사용자가 결정):

```text
# judge는 생성 모델보다 강한 모델 권장. 비우면 LLM_MODEL 사용
JUDGE_MODEL=
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/pytest tests/eval/test_judge.py -v`
Expected: 3 passed

Run (Task 9 스모크 결과로): `.venv/bin/python -m eval.run_judge --e2e eval/results/_smoke_e2e.json --out eval/results/_smoke_judge.json --sheet ""`
Expected: 장소별 `ok 4` 형태 출력과 요약 dict

- [ ] **Step 5: 커밋**

```bash
git add eval/judge.py eval/run_judge.py eval/judge_agreement.py tests/eval/test_judge.py
git commit -m "feat(eval): 리뷰 근거 LLM-as-judge와 수기 라벨 일치도(가중 kappa)"
```

---

### Task 11: 베이스라인 측정, 리포트, 태그

**Files:**
- Create: `eval/report.py`, `docs/eval/baseline.md`(생성물), `eval/results/*.json`(생성물)
- Test: `tests/eval/test_report.py`

**Interfaces:**
- Consumes: 세 결과 JSON의 `summary`/`mean_*` 필드
- Produces: `render(retrieval: dict, e2e: dict, judge: dict, agreement: dict | None) -> str`, CLI `python -m eval.report --tag baseline`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/eval/test_report.py`:

```python
from eval.report import render

R = {"llm_model": "m", "k": 5, "summary": {"all": {"n": 2, "region_acc": 0.5, "category_acc": 1.0,
      "region_hit1": 0.5, "diversity": 1.0, "empty": 0.0, "llm_calls": 3.0, "latency_s": 1.0},
      "gwangju": {"n": 1, "region_acc": 0.0, "category_acc": 1.0, "region_hit1": 0.0,
                  "diversity": 1.0, "empty": 0.0, "llm_calls": 3.0, "latency_s": 1.0}}}
E = {"llm_model": "m", "summary": {"all": {"n": 1, "ok": 1.0, "slot_fill": 0.8, "category_adherence": 0.6,
      "region_acc": 0.9, "dup_rate": 0.0, "overflow": 0.0, "total_move_min": 50.0, "llm_calls": 31.0,
      "input_tokens": 1000.0, "output_tokens": 100.0, "maps_calls": 12.0, "latency_s": 90.0, "n_places": 5.0}}}
J = {"judge_model": "j", "mean_score": 3.5, "mean_trip_score": 3.4,
     "status_counts": {"ok": 4, "unmatched": 1, "parse_error": 0}}


def test_render_contains_key_numbers():
    md = render(R, E, J, {"n": 40, "quadratic_weighted_kappa": 0.71})
    assert "| gwangju | 1 | 0.00 |" in md
    assert "31.0" in md and "3.50" in md and "0.71" in md


def test_render_without_agreement():
    assert "수기 라벨 미완료" in render(R, E, J, None)
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/pytest tests/eval/test_report.py -v`
Expected: FAIL (`ModuleNotFoundError: eval.report`)

- [ ] **Step 3: 구현**

`eval/report.py`:

```python
import argparse
import json
from pathlib import Path


def _f(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def render(retrieval: dict, e2e: dict, judge: dict, agreement: dict | None) -> str:
    out = ["# 베이스라인 평가 결과", "",
           f"- 생성 모델: `{e2e['llm_model']}` / judge: `{judge['judge_model']}`", ""]
    out += ["## A. 검색 컴포넌트 (k={})".format(retrieval["k"]), "",
            "| 그룹 | n | 지역 정확도@k | 지역 Hit@1 | 카테고리 정확도@k | 다양성@k | 빈 결과 비율 | LLM 호출 |",
            "|---|---|---|---|---|---|---|---|"]
    for g, s in retrieval["summary"].items():
        out.append(f"| {g} | {s['n']} | {_f(s['region_acc'])} | {_f(s['region_hit1'])} | "
                   f"{_f(s['category_acc'])} | {_f(s['diversity'])} | {_f(s['empty'])} | {_f(s['llm_calls'], 1)} |")
    s = e2e["summary"]["all"]
    out += ["", "## B. E2E 일정 생성", "", "| 지표 | 값 |", "|---|---|"]
    for label, key, d in [("성공률", "ok", 2), ("슬롯 충족률", "slot_fill", 2),
                          ("카테고리 준수율", "category_adherence", 2), ("일정 지역 정확도", "region_acc", 2),
                          ("중복률", "dup_rate", 2), ("초과 배정 수", "overflow", 1),
                          ("총 이동 시간(분)", "total_move_min", 1), ("LLM 호출 수", "llm_calls", 1),
                          ("입력 토큰", "input_tokens", 0), ("출력 토큰", "output_tokens", 0),
                          ("Maps 호출 수", "maps_calls", 1), ("소요 시간(초)", "latency_s", 1)]:
        out.append(f"| {label} | {_f(s.get(key), d)} |")
    c = judge["status_counts"]
    out += ["", "## C. 취향 적합도 (LLM-as-judge, 1~5)", "",
            f"- 장소 평균: **{_f(judge['mean_score'])}**, 일정 평균: {_f(judge['mean_trip_score'])}",
            f"- 채점 {c['ok']}건 / 메타데이터 불일치 {c['unmatched']}건 / 파싱 실패 {c['parse_error']}건"]
    if agreement:
        out.append(f"- judge 신뢰도: 수기 라벨 {agreement['n']}건과 가중 kappa **{_f(agreement['quadratic_weighted_kappa'])}**")
    else:
        out.append("- judge 신뢰도: 수기 라벨 미완료")
    return "\n".join(out) + "\n"


def main(args):
    load = lambda p: json.loads(Path(p).read_text())
    agreement = load(args.agreement) if args.agreement and Path(args.agreement).exists() else None
    md = render(load(f"eval/results/retrieval_{args.tag}.json"), load(f"eval/results/e2e_{args.tag}.json"),
                load(f"eval/results/judge_{args.tag}.json"), agreement)
    Path(f"docs/eval/{args.tag}.md").parent.mkdir(parents=True, exist_ok=True)
    Path(f"docs/eval/{args.tag}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--agreement", default="eval/results/agreement_baseline.json")
    main(ap.parse_args())
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/pytest tests/eval/test_report.py -v && .venv/bin/pytest -m "not slow" -q`
Expected: 2 passed, 이어서 전체 빠른 테스트 모두 통과

- [ ] **Step 5: 개발 모델(Ollama)로 전체 리허설**

```bash
.venv/bin/python -m eval.run_retrieval
.venv/bin/python -m eval.run_e2e
.venv/bin/python -m eval.run_judge
.venv/bin/python -m eval.report --tag baseline
```
Expected: `docs/eval/baseline.md` 생성, 세 섹션 모두 숫자로 채워짐. 수치 이상(예: 모든 지역 정확도 0) 시 스모크 결과의 `top`·`target_region` 필드로 원인 확인.

- [ ] **Step 6: 수기 라벨 (사용자)**

`eval/data/judge_labels.csv`의 `human_score` 열 40칸을 1~5로 채운다(judge 점수는 시트에 없음). 이후:

```bash
.venv/bin/python -m eval.judge_agreement > eval/results/agreement_baseline.json
```

- [ ] **Step 7: 공식 측정 (OpenAI)**

`.env`에서 `LLM_MODEL=openai:gpt-5-mini`, `OPENAI_API_KEY`, `JUDGE_MODEL` 설정 후:

```bash
.venv/bin/python -m eval.run_retrieval --no-cache
.venv/bin/python -m eval.run_e2e --no-cache
.venv/bin/python -m eval.run_judge --no-cache --sheet ""
.venv/bin/python -m eval.report --tag baseline
```

Ollama 리허설 결과를 보존하려면 먼저 `eval/results/*_baseline.json`을 `*_baseline_ollama.json`으로 이름을 바꾼다.

- [ ] **Step 8: 커밋과 태그**

```bash
git add eval/report.py tests/eval/test_report.py eval/results/*_baseline*.json eval/data/judge_labels.csv docs/eval/baseline.md .env.example
git commit -m "eval: 베이스라인 측정 결과 (검색·E2E·judge)"
git tag baseline-v0
```

`.env.example`은 `.env`에서 값만 비운 사본(`sed -E 's/=.*/=/' .env > .env.example`).

---

## Self-Review 결과

- 스펙 커버리지: 4장 A층(Task 3·8), B층(Task 4·9), 지표 표(Task 6), Maps 캐시(Task 5), judge와 검증(Task 10), 3.6 LLM 설정 중 P1 범위(Task 5 `make_llm`), 6장 Routes API 결정(Task 5). 3장 목표 아키텍처는 P3 범위라 제외.
- 스펙과 다른 점: 스펙 4장은 "평가 시 날씨 고정 응답"이었으나 OWM 키가 확인되어 실제 API 사용(날씨는 채점 대상 아님) — 계획이 단순한 쪽.
- 알려진 한계(리포트에 명시): Routes API 전환으로 `departure_time` 무시, 옛 코드의 퍼지 로직은 그대로라 결과에 LLM 비결정성 포함(캐시로 리허설 재현성 확보, 공식 측정은 1회).
