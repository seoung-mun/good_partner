# castone_3 리팩토링 설계

- 작성일: 2026-09-28
- 상태: 초안 (사용자 리뷰 대기)
- 참고: LangChain v1 / LangGraph 공식 문서(workflows-agents, agents, tools, structured-output, middleware, graph-api), `kakaotechcampus-4/chonnam-clone`

## 1. 목표

| 구분 | 내용 |
|---|---|
| 목적 | 취업용 — "RAG 파이프라인을 측정하고 개선했다"를 before→after 수치로 설명 |
| 베이스라인 | 기존 코드의 검색 로직을 **수정 전 그대로** 박제해 측정 |
| 설계 원칙 | 초안 생성은 **LangGraph 워크플로우**, 사용자 수정은 **tool calling 에이전트**. 각 노드 내부는 **LangChain 컴포넌트(LCEL, Retriever, structured output)** 로 구성 |

비목표: 크롤링 재수행, UI 디자인 개편, 배포, 멀티에이전트(Supervisor)·Deep Agents 도입(P3 이후 재검토).

### 성공 지표

| 지표 | 측정 방법 | 단계 |
|---|---|---|
| 지역 정확도@k | 골든셋 쿼리 → Top-k 결과의 `지역`이 정답 시군구와 일치하는 비율 | P1~ |
| 카테고리 정확도@k | Top-k 결과의 `카테고리` 일치 비율 | P1~ |
| 장소 다양성@k | Top-k 내 고유 장소 수 / k | P1~ |
| 일정 1회 생성당 LLM 호출 수·소요 시간 | callback trace 집계 | P3 |
| 취향 적합도 | LLM-as-judge (근거 리뷰 기반) | P4 |

## 2. 현재 상태 진단 (AS-IS)

```
entry_router(키워드) → PlannerAgent | EditorAgent  (같은 tool 7개 바인딩)
        → agent_router → call_tools_node(앵커 추론·ban list·차단·퍼지 삭제·삽입·타임라인·PDF)
        → route_after_tools → Planner | Editor | END (루프)
```

| # | 문제 | 근거 |
|---|---|---|
| 1 | 결정적 규칙(일차별 슬롯 수·순서)을 LLM 루프가 1곳씩 채움 → 재귀 감지·`[SYSTEM ALERT]` 주입 등 땜질 | `graph_flow.py` planner_prompt, agent_router, execute_tools |
| 2 | `find_and_select_best_place`가 9단계 수행, 장소 1곳당 LLM 3회 | `tools.py` |
| 3 | `delete_place`/`replace_place`는 입력 echo, 실제 삭제는 difflib 유사도>0.5 퍼지 매칭 | `process_deletions` |
| 4 | tool 입출력 무스키마, `itinerary: List[Dict]`, 실패가 정상 결과로 섞임 | 전반 |
| 5 | **풀링 불일치**: 문서=CLS(노트북), 쿼리=Mean(`HuggingFaceEmbeddings`) — 코사인 0.86~0.88 | 2026-09-28 검증 |
| 6 | **광주 필터 버그**: 접미사 제거 후 "광주"만 비교 → 광주광역시↔경기도 광주시 상호 통과 | `_filter_candidates` 검증 |
| 7 | Top-k를 같은 장소 리뷰 chunk가 점유(리뷰 단위 인덱싱, 장소 단위 dedupe는 검색 후) | 구조 |
| 8 | 리뷰 추출 코드 미동작(`page_content`에 줄바꿈 없음), `exclude_places=[]` 가변 기본값, `config.py` 함수 중복, PDF tool에 없는 인자 전달, `scheduler`↔`tools` 순환 import | 코드 |

## 3. 목표 아키텍처 (TO-BE)

### 3.1 디렉터리

```
castone_3/
├─ app.py, pages/                 UI만: 폼 → graph.invoke, 결과 렌더링
├─ src/
│  ├─ domain/                     LangChain 비의존 순수 로직
│  │   models.py                  Pydantic 스키마 (3.3)
│  │   region.py                  지역 정규화 (광주 수정 위치)
│  │   schedule.py                슬롯 템플릿, 타임라인 계산
│  ├─ retrieval/
│  │   store.py                   FAISS 로드 (임베딩 설정 단일화)
│  │   retriever.py               PlaceRetriever(BaseRetriever)
│  ├─ services/                   maps.py, weather.py, pdf.py (외부 API 래퍼)
│  └─ agent/
│      state.py, prompts.py
│      chains.py                  LCEL 체인 (parse, briefing)
│      tools.py                   수정 에이전트용 얇은 tool
│      graph.py                   워크플로우 + 에이전트 조립
├─ eval/
│   golden.jsonl
│   baseline_retriever.py         기존 _search_docs + _filter_candidates 원본 복사
│   run_eval.py
└─ scripts/reembed.py
```

의존 방향: `pages → agent → (retrieval, services) → domain`. domain은 아무것도 import하지 않음 (chonnam-clone `fixed/schedule_decision.py` 패턴).

### 3.2 LangGraph 구조

```
START
  │
route_entry  (결정적: state.itinerary 유무로 분기, 키워드 매칭 폐기)
  ├─ 없음 ─▶ [초안 생성 워크플로우]
  │          parse_request
  │            → build_slots
  │            → (Send ×N) retrieve_candidates     ← 슬롯별 병렬
  │            → assemble_itinerary
  │            → plan_routes ∥ fetch_weather       ← 병렬
  │            → compose_briefing → END
  │
  └─ 있음 ─▶ [수정 에이전트]
             edit_agent (create_agent 서브그래프, tools는 3.5)
               → plan_routes → compose_briefing → END
```

대화 상태는 Streamlit `session_state`에 통째로 들고 다니지 않고 `checkpointer`(`InMemorySaver`) + `thread_id`로 유지.

### 3.3 도메인 모델 (`src/domain/models.py`)

```python
class Region(BaseModel):
    sido: str                     # "경기도"
    sigungu: str | None           # "광주시"
    source: Literal["alias", "maps", "llm"]

class TripRequest(BaseModel):     # parse_request 출력
    destination: Region
    start_date: date
    days: int = Field(ge=1, le=7)
    group_type: str
    style: list[str]
    preference_keywords: list[str] = Field(description="리뷰 검색에 쓸 취향 키워드")
    avoid: list[str] = []

class Slot(BaseModel):
    slot_id: str                  # "d1-s2"
    day: int
    order: int
    category: Literal["식당", "카페", "관광지"]
    meal: Literal["점심", "저녁"] | None = None

class PlaceCandidate(BaseModel):
    place_id: str                 # f"{지역}|{장소명}" — 동명 장소 구분
    name: str
    region: str
    category: str
    rating: float
    address: str
    evidence_reviews: list[str]   # 근거 리뷰 (최대 3)
    score: float

class ItineraryItem(BaseModel):
    slot: Slot
    place: PlaceCandidate
    start_time: time | None = None
    move_from_prev: "RouteLeg | None" = None
```

### 3.4 노드별 내부 구성 (LangChain 활용)

| 노드 | 종류 | 내부 구성 | LLM 호출 |
|---|---|---|---|
| `parse_request` | 체인 | `ChatPromptTemplate \| llm.with_structured_output(TripRequest)` | 1 |
| `build_slots` | 코드 | `domain.schedule.build_slots(days)` | 0 |
| `retrieve_candidates` | Retriever | `PlaceRetriever.ainvoke(query)` — `BaseRetriever` 서브클래스: 멀티쿼리(템플릿) → `similarity_search(fetch_k)` → `region`/`category` 필터 → 장소 단위 dedupe → `PlaceCandidate[]` | 0 |
| `assemble_itinerary` | 코드 | 일차별로 직전 장소와의 거리 + 검색 점수로 선택, 슬롯 간 중복 배제 | 0 |
| `plan_routes` | 코드 | `services.maps` 거리행렬 → 순서 최적화 → `domain.schedule.timeline` | 0 |
| `fetch_weather` | 코드 | `services.weather` | 0 |
| `compose_briefing` | 체인 | `prompt \| llm.with_structured_output(Briefing)` — 근거 리뷰만 사용, 장소별 추천 사유 포함 | 1 |
| `edit_agent` | 에이전트 | `create_agent(model, tools, system_prompt, state_schema=TripAgentState, middleware=[...])` | 가변 |

초안 생성 LLM 호출: 기존 장소당 3회(10곳 기준 30회 이상) → 2회 (+ 지역 해석 실패 시 1회).

`PlaceRetriever`를 `BaseRetriever`로 만드는 이유: Runnable이라 LCEL에 합성되고, P4에서 `EnsembleRetriever`(BM25 hybrid), `ContextualCompressionRetriever` + `CrossEncoderReranker`로 감싸기만 하면 됨.

### 3.5 수정 에이전트 tool

공통 규칙 (chonnam-clone + LangChain tools 문서):
- 모든 tool은 `@tool(args_schema=...)` + Pydantic `Field(description=...)`.
- 상태 변경 tool은 `ToolRuntime`으로 state를 읽고 `Command(update={"itinerary": ..., "messages": [ToolMessage(...)]})`로 반영 — `call_tools_node`의 수동 상태 조작을 대체.
- 조회 tool 반환은 envelope: `{"ok": bool, "tool_name": str, "data": ..., "error": str | None}` (`json.dumps(..., ensure_ascii=False)`).
- 장소 지정은 이름이 아닌 `slot_id` → difflib 퍼지 매칭 제거.

| tool | 입력 | 동작 |
|---|---|---|
| `search_places` | region, category, query, k | PlaceRetriever 조회, 후보 반환 (상태 변경 없음) |
| `replace_slot` | slot_id, place_id | 슬롯의 장소 교체 |
| `remove_slot` | slot_id | 슬롯 삭제 |
| `add_slot` | day, after_slot_id, category, place_id | 슬롯 추가 |
| `reorder_day` | day | 해당 일차 경로 순서 최적화 |
| `export_pdf` | – | PDF 생성 플래그 |

미들웨어 (built-in):
- `ModelCallLimitMiddleware(run_limit=...)`, `ToolCallLimitMiddleware` — 기존 재귀 감지 코드 대체
- `ToolRetryMiddleware` — Maps/날씨 API 일시 오류
- `ToolErrorMiddleware` — 예외를 ToolMessage로 변환해 에이전트가 인지

### 3.6 LLM 설정 (provider 교체 가능)

모든 LLM은 `init_chat_model(settings.llm_model)` 한 곳에서 생성. 코드에 provider를 하드코딩하지 않는다.

| 프로필 | `LLM_MODEL` | 용도 |
|---|---|---|
| dev | `ollama:qwen2.5:7b-instruct-q4_K_M` | 개발·스모크 테스트, 무비용 |
| eval/prod | `openai:gpt-5-mini` | 공식 지표 측정, 데모 |

- 임베딩은 LLM provider와 무관하게 로컬 `upskyy/bge-m3-korean` 유지 (OpenAI 임베딩으로 바꾸면 재임베딩·비용 발생).
- OpenAI structured output은 `ProviderStrategy(strict=True)`가 기본 선택됨 → strict 스키마 제약(기본값 필드는 `Optional` 처리) 고려해 스키마 작성.
- 소형 모델은 tool calling·structured output 실패율이 높음 → 2B급은 쓰지 않고 7B instruct 사용. 실패는 `ToolStrategy(handle_errors=True)`/`ModelRetryMiddleware`로 흡수.
- **공식 지표의 before/after는 반드시 같은 모델로 측정**한다. 베이스라인(기존 로직)도 Gemini 대신 동일 모델로 교체해 실행 — 비교 대상은 "로직"이지 "모델"이 아니므로.

설정은 `src/config.py`의 `Settings`(pydantic-settings, `.env`)로 단일화: `LLM_MODEL`, `OPENAI_API_KEY`, `GMAPS_API_KEY`, `OWM_API_KEY`, `INDEX_DIR`, `EMBED_MODEL`, `EMBED_POOLING`.

### 3.7 관측

LangChain callback으로 노드별 LLM 호출 수·토큰·지연 시간 수집 (chonnam-clone `fixed/trace.py` 패턴). P3 지표의 근거.

## 4. 평가 설계 (P1)

두 층으로 측정한다. 컴포넌트 평가만으로 한정하지 않는다.

**A. 검색 컴포넌트 평가** — 원인 분석용, 빠름, LLM 불필요
**B. E2E 일정 평가** — 실제 사용자 경험, 여행 요청 → 완성 일정

E2E 지표 (골든 여행 요청 20~30개):

| 지표 | 정의 |
|---|---|
| 일정 지역 정확도 | 일정에 들어간 장소 중 목적지 시군구 소속 비율 |
| 카테고리 준수율 | 슬롯 카테고리(식당/카페/관광지)와 배정 장소 일치 비율 |
| 슬롯 충족률 | 템플릿 슬롯 중 실제 장소가 채워진 비율 ("추천 장소 없음" 제외) |
| 중복률 | 한 일정 내 같은 `place_id` 중복 |
| 총 이동 시간 | Maps 기준 일차별 이동 시간 합 |
| LLM 호출 수·토큰·소요 시간 | callback trace |

- E2E 실행 조건: Google Maps 키(발급 완료), LLM(`LLM_MODEL`). 날씨는 장소 품질에 영향이 없으므로 평가 시 고정 응답으로 대체 → OWM 키 없이 실행 가능.
- Maps 호출은 요청 단위로 디스크 캐시 → 반복 평가 비용·결과 변동 제거.
- 베이스라인 E2E: 기존 `graph_flow`를 LLM만 `init_chat_model`로 교체해 실행 (3.6).

**A층 골든셋**
- `eval/golden.jsonl`: 메타데이터(254 시군구 × 3 카테고리)에서 생성, 60~100개.
  - 동명·유사 지역 의도적 포함: 광주광역시/경기도 광주시, 부산·서울·대구·인천 중구·동구·서구 등, 고성군(강원/경남)
  - 약칭·지명 쿼리: "해운대 맛집", "광안리 카페"
  - 필드: `query, destination, category, gold_sido, gold_sigungu`
- 채점: 메타데이터 `지역`·`카테고리`와 자동 비교 (라벨링 불필요).
- 베이스라인: `eval/baseline_retriever.py`에 기존 `_search_docs` + `_filter_candidates`를 원본 그대로 복사, LLM 단계(지역 정규화·멀티쿼리)는 LLM 없이 재현 가능한 부분만 사용하고 어떤 부분을 고정했는지 문서화.
- 인덱스: `review_faiss/`(CLS, 원본 재현) — 베이스라인, `review_faiss_mean/`(Mean) — 풀링 수정본.

## 5. 단계 계획

| 단계 | 내용 | 산출 지표 |
|---|---|---|
| P0 | 인덱스 재생성 (`scripts/reembed.py`, 진행 중) | – |
| P1 | 골든셋 + 베이스라인 측정 | 지역·카테고리 정확도@k, 다양성@k |
| P2 | 버그 수정, 건별 재측정: ① 풀링 ② 광주 필터 ③ 장소 dedupe ④ 기타(#8) | 버그별 before→after |
| P3 | 3장 구조로 재작성 (domain → retrieval → services → agent → UI 순) | LLM 호출 수, 생성 시간 |
| P4 | 검색 고도화: 메타데이터 pre-filter(지역별 인덱스 또는 Qdrant), BM25 hybrid, reranker, MMR | 정확도@k, 취향 적합도 |
| P5 | 테스트, README, 아키텍처 다이어그램, 포트폴리오 수치 정리 | – |

## 6. 결정 사항 / 미결

- 결정: 초안=워크플로우, 수정=에이전트 (2026-09-28 합의)
- 결정: 원본 크롤링 CSV 불필요 — `metadata_list.pkl`이 임베딩 입력 전체
- 결정: LLM은 OpenAI(`gpt-5-mini`)로 전환, 개발·테스트는 Ollama(`qwen2.5:7b-instruct`) — `init_chat_model`로 설정만 교체 (2026-09-28)
- 결정: 평가는 검색 컴포넌트(A) + E2E(B) 두 층 모두 수행
- 의존성: `langchain>=1.0`, `langchain-openai`, `langchain-ollama`, `pydantic-settings` 추가, `langchain_google_genai` 제거
- 준비물: `.env`에 `GMAPS_API_KEY`(발급 완료), `OPENAI_API_KEY`(공식 측정 전까지)
- 결정: Google Maps는 Routes API(`computeRoutes`, `computeRouteMatrix`)로 이전 — Directions·Distance Matrix는 legacy라 신규 프로젝트에서 활성화 불가(2026-09-28 키 검증 시 REQUEST_DENIED). 베이스라인 E2E도 Maps 어댑터만 Routes API로 교체해 실행(LLM 교체와 같은 원칙)
- 주의: 16GB RAM — Ollama 7B(~5GB)는 재임베딩 종료 후 실행
