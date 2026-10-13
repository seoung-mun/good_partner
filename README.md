# Castone 3

실제 방문자 리뷰를 근거로 동선까지 고려한 여행 일정을 만들어 주는 AI 여행 플래너입니다. 목적지·날짜·동행·취향을 입력하면 리뷰 벡터 검색으로 장소를 고르고 Google Maps로 이동 시간을 계산해 일정표를 만들고, 채팅으로 수정한 뒤 PDF로 내려받을 수 있습니다.

## 구조

```text
app.py                     # 여행 정보 입력 (시작 페이지)
pages/
├── trip_planner.py        # 일정 생성·카드·지도·수정 채팅·PDF 다운로드
└── review_writer.py       # 리뷰 작성 → 일정 수 이상 쌓이면 인덱스 반영
src/
├── config.py              # 환경변수, get_llm(), Maps 클라이언트, 인덱스 경로
├── llm_factory.py         # LLM_MODEL("provider:model") → chat model
├── errors.py              # AppError 계층 (code + 사용자 안내 문구)
├── agent/
│   ├── workflow.py        # 일정 생성 LangGraph (고정 워크플로)
│   ├── edit_agent.py      # 일정 수정 tool-calling 에이전트
│   ├── edit_tools.py      # replace_slot / remove_slot / add_slot
│   ├── chains.py          # 취향 키워드 추출, 근거 리뷰 기반 브리핑
│   └── render.py          # 일정 → 마크다운
├── domain/                # 순수 도메인 (모델, 지역 해석, 시간표·배정)
├── retrieval/             # FAISS 로드, 지역·카테고리 선필터 검색, 시간대 적합도
├── services/              # Google Maps, OpenWeatherMap
├── ui/                    # 카드·지도·텍스트·PDF
└── rag_updater.py         # 누적 리뷰 → 검증된 새 인덱스로 교체
eval/                      # 골든셋 기반 E2E·수정·검색 평가 (docs/eval/ 결과)
scripts/reembed.py         # 원본 메타데이터로 FAISS 인덱스 재생성
```

### 일정 생성 그래프 (`src/agent/workflow.py`)

```text
START → resolve_region ┬→ extract_preferences → search → locate ┬→ plan_routes ┬→ render → END
                       └→ fetch_weather ──────────────────────────┴→ brief ──────┘
```

- LLM은 `extract_preferences`, `brief` 두 노드에서만 호출합니다 (요청당 2회). 나머지는 결정적 코드입니다.
- 각 단계의 실패는 예외 대신 `StepError`로 상태에 쌓이고, 화면 상단에 안내로 표시됩니다.
- 수정은 예측할 수 없는 요청이라 별도의 tool-calling 에이전트(`edit_agent.py`)가 처리하고, 끝나면 시간표를 다시 계산합니다.
- 리뷰·장소명은 사용자가 쓴 글이라 프롬프트에 `<review>`/`<place>` 태그로 감싸 자료로만 다룹니다.

## 실행

### 1. 의존성

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-eval.txt   # 앱 + 테스트 (faiss, sentence-transformers, pytest)
```

### 2. 환경 변수 (`.env`)

| 키 | 용도 | 필수 |
| --- | --- | --- |
| `LLM_MODEL` | `provider:model` 형식 (예: `openai:gpt-4o-mini`, `ollama:qwen2.5:7b-instruct-q4_K_M`). provider별 API 키도 함께 설정 | 필수 |
| `GMAPS_API_KEY` | 좌표·경로 계산 | 필수 (없으면 동선 정보가 빠짐) |
| `OWM_API_KEY` | 날씨 요약 | 선택 |
| `INDEX_DIR` | 인덱스 폴더 이름 (기본 `review_faiss_mean`) | 선택 |
| `EMBED_DEVICE` | 임베딩 장치 (`cpu` 기본, 개발 시 `mps`/`cuda`) | 선택 |
| `LLM_TIMEOUT_S`, `LOG_LEVEL`, `MAPS_CACHE`, `MAPS_CACHE_DIR` | 호출 제한 시간, 로그 레벨, Maps 응답 캐시 | 선택 |

### 3. 리뷰 벡터 인덱스

앱은 프로젝트 루트의 `review_faiss_mean/`(LangChain FAISS 형식, Mean 풀링)을 읽습니다. 저장소에는 포함되어 있지 않습니다.

- 원본 메타데이터(`metadata_list.pkl`)가 있으면 `python scripts/reembed.py`로 다시 만듭니다 (`docs/refactor-log.md` P0 참고).
- 공유 링크의 구형 파일:
  - https://drive.google.com/file/d/1VH2m0IzGnyajpLCS0oUcWAZmtMV4Ktui/view?usp=drive_link
  - https://drive.google.com/file/d/1A2bc5mO-kNOQl6N_b20M7Z0yuPfFjmPF/view?usp=drive_link

인덱스를 불러오지 못하면 일정 페이지가 원인과 함께 오류를 표시하고 멈춥니다.

### 4. 앱 실행

```bash
streamlit run app.py
```

## 리뷰 반영

`pages/review_writer.py`에서 쓴 리뷰는 `new_reviews.csv`에 쌓이고, 10개 이상이면 `src/rag_updater.py`가 반영합니다.

1. 원본 리뷰를 `reviews_log.csv`에 추가만 합니다 (인덱스 재생성용 원본).
2. 기존 인덱스에 리뷰를 더해 `<인덱스>.staging-<시각>`에 저장하고, 다시 읽어 벡터·문서 수를 검증합니다.
3. 검증을 통과하면 기존 인덱스를 `<인덱스>.prev-<시각>`으로 보관하고 새 인덱스로 교체합니다.

실패하면 `IndexUpdateError`가 나고 기존 인덱스와 `new_reviews.csv`는 그대로 남습니다.

## 테스트와 평가

```bash
pytest -m "not slow"     # 인덱스·모델 없이 도는 단위 테스트
pytest                   # 실제 인덱스 무결성 테스트 포함
```

평가 스크립트와 결과는 `eval/`, `docs/eval/`에 있고, 리팩토링 전후 비교는 `docs/eval/final.md`, 단계별 기록은 `docs/refactor-log.md`에 있습니다.

## 참고

- 날씨는 OpenWeatherMap 5일 예보라 먼 날짜는 요약이 빠집니다.
- PDF는 루트의 `NanumGothic.ttf`, `NanumGothicBold.ttf`가 필요합니다.
