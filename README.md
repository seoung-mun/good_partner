# Castone 3

Streamlit, LangGraph, Gemini, Google Maps API를 조합한 AI 여행 플래너입니다. 사용자가 목적지, 일정, 동행 형태, 선호 스타일을 입력하면 리뷰 기반 장소 검색과 경로 최적화를 거쳐 여행 일정을 만들고, 이후 채팅으로 수정한 뒤 PDF로 내려받을 수 있습니다.

## 주요 기능

- 목적지, 출발지, 날짜, 여행 성격을 입력받아 여행 계획 시작
- 리뷰 벡터 DB(FAISS) 기반 장소 검색
- LangGraph 에이전트를 통한 일정 생성 및 수정
- Google Maps 기반 좌표 조회, 이동 경로 계산, 순서 최적화
- OpenWeatherMap 기반 날씨 요약
- 일정 PDF 생성 및 다운로드
- 사용자 리뷰 누적 후 벡터 DB 자동 업데이트

## 기술 스택

- Python
- Streamlit
- LangGraph / LangChain
- Google Gemini (`langchain_google_genai`)
- Google Maps API
- OpenWeatherMap API
- FAISS

## 프로젝트 구조

```text
.
├── app.py                     # 여행 정보 입력 시작 페이지
├── pages/
│   ├── trip_planner.py        # 채팅형 일정 생성/수정 + PDF 다운로드
│   └── review_writer.py       # 리뷰 입력 및 벡터 DB 갱신
├── src/
│   ├── config.py              # LLM, Google Maps, FAISS 로더
│   ├── graph_flow.py          # LangGraph 에이전트 흐름
│   ├── tools.py               # 장소 검색, 경로 계산, 날씨 조회 도구
│   ├── rag_updater.py         # 새 리뷰를 벡터 DB에 반영
│   ├── scheduler/             # 일차별 일정 배치 로직
│   └── region_cut_fuzz.py     # 지역명 정규화
├── NanumGothic.ttf            # PDF 한글 폰트
├── NanumGothicBold.ttf        # PDF 한글 볼드 폰트
├── new_reviews.csv            # 누적 리뷰 임시 저장 파일
└── requirements.txt
```

## 사전 준비

### 1. 의존성 설치

가상환경 사용을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`uv`를 사용한다면 아래처럼 설치해도 됩니다.

```bash
pip install uv
uv pip install -r requirements.txt
```

### 2. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 만들고 아래 값을 설정하세요.

```env
GMAPS_API_KEY=your_google_maps_api_key
OWM_API_KEY=your_openweathermap_api_key
GOOGLE_API_KEY=your_gemini_api_key
```

- `GMAPS_API_KEY`: 경로 계산, 좌표 변환, 거리행렬 계산에 사용
- `OWM_API_KEY`: 여행 날짜 날씨 조회에 사용
- `GOOGLE_API_KEY`: `ChatGoogleGenerativeAI`가 Gemini 호출 시 사용

### 3. 리뷰 벡터 DB 준비

이 프로젝트는 루트의 `review_faiss/` 디렉터리를 읽어 장소 리뷰 벡터 DB를 로드합니다. 저장소에는 포함되어 있지 않으므로 직접 준비해야 합니다.

1. 프로젝트 루트에서 디렉터리를 생성합니다.

```bash
mkdir -p review_faiss
```

2. 아래 공유 링크의 파일을 받아 `review_faiss/` 안에 넣습니다.

- https://drive.google.com/file/d/1VH2m0IzGnyajpLCS0oUcWAZmtMV4Ktui/view?usp=drive_link
- https://drive.google.com/file/d/1A2bc5mO-kNOQl6N_b20M7Z0yuPfFjmPF/view?usp=drive_link

주의:

- 현재 앱은 `src/config.py`에서 LangChain 형식의 FAISS 로컬 인덱스를 기대합니다.
- 다운로드한 파일이 구형 `faiss.index` / `metadata_list.pkl` 형식이라면 `src/convert_faiss_format.py`를 경로에 맞게 수정한 뒤 변환이 필요할 수 있습니다.

## 실행

```bash
streamlit run app.py
```

실행 후 기본 진입 페이지는 여행 정보 입력 화면입니다.

- 메인 시작 페이지: `app.py`
- 일정 생성/수정 페이지: `pages/trip_planner.py`
- 리뷰 작성 페이지: `pages/review_writer.py`

## 사용 흐름

### 일정 생성

1. 목적지, 출발지, 날짜를 입력합니다.
2. 모임 성격과 선호 스타일을 선택합니다.
3. 상세 취향을 자유롭게 적습니다.
4. `AI 플래너에게 정보 전달하고 시작하기`를 누릅니다.

초기 일정이 생성된 뒤에는 채팅으로 아래와 같이 수정할 수 있습니다.

- `근처 관광지 추천해줘`
- `점심 식당 바꿔줘`
- `카페 하나 추가해줘`
- `경로 최적화해줘`
- `PDF로 만들어줘`

### 리뷰 누적 및 DB 업데이트

`pages/review_writer.py`에서는 장소 리뷰를 직접 입력할 수 있습니다.

- 리뷰는 우선 `new_reviews.csv`에 누적됩니다.
- 리뷰가 10개 이상 쌓이면 `src/rag_updater.py`가 벡터 DB 업데이트를 시도합니다.
- 업데이트가 끝나면 `new_reviews.csv`는 비워집니다.

## 참고 사항

- `pages/trip_planner.py`는 진입 시 FAISS DB를 먼저 로드합니다. `review_faiss/`가 없으면 추천 품질이 크게 떨어지거나 기능 일부가 동작하지 않을 수 있습니다.
- 날씨 조회는 OpenWeatherMap 5일 예보를 사용하므로 먼 미래 일정에는 한계가 있습니다.
- PDF 생성은 `NanumGothic.ttf`, `NanumGothicBold.ttf`를 사용하므로 폰트 파일을 유지해야 합니다.
- `src/config.py`에는 중복 정의가 일부 포함되어 있습니다. 현재 README는 동작 기준으로 정리했습니다.
