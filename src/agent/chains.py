"""LLM 프롬프트 체인: 지역명 정규화, 멀티쿼리 생성, 장소 추천사."""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.config import LLM

query_gen_prompt = PromptTemplate.from_template("""
역할: 당신은 '검색어 최적화 전문가'입니다.
목표: 사용자의 요청과 취향을 분석하여, 벡터 데이터베이스에서 가장 정확한 장소를 찾을 수 있는 **3개의 검색 쿼리**를 생성하세요.

[입력 정보]
- 여행지/지역: {target_region}
- 사용자 검색어: {query}
- 사용자 취향/정보: {user_info}
- 카테고리 필터: {category_filter}

[지침]
1. 사용자의 자연어 문장(취향)에서 **핵심 키워드(형용사, 명사)**만 추출하세요. (예: "조용한", "뷰맛집", "재즈")
2. 지역명과 핵심 키워드를 조합하여 검색어를 만드세요.
3. 다음 3가지 관점의 쿼리를 생성하세요:
   - 쿼리 1: 지역명 + 사용자 검색어 (기본 정확도 중심)
   - 쿼리 2: 지역명 + 사용자 검색어 + 취향 키워드 (구체적 니즈 중심)
   - 쿼리 3: 지역명 + 분위기/테마 키워드 (광범위 탐색)
4. 결과는 오직 쉼표(,)로 구분된 문자열로만 출력하세요. 다른 설명은 생략하세요.

[예시]
입력: 지역="서울", 검색어="카페", 취향="조용하고 작업하기 좋은 곳", 필터="카페"
출력: 서울 카페, 서울 조용한 작업하기 좋은 카페, 서울 스터디 카페 분위기
""")

query_gen_chain = query_gen_prompt | LLM | StrOutputParser()


region_prompt = PromptTemplate.from_template("""
역할: 당신은 '지명 정규화 전문가'입니다.
목표: 사용자의 검색어("{query}")와 여행 목적지("{destination}")를 보고, 검색 대상이 되는 **정확한 행정구역 명칭** 하나만 출력하세요.

[규칙]
1. 검색어에 '해운대', '송도' 같은 구체적 지명이 있다면, 해당 지명의 **공식 행정구역명**을 찾으세요.
2. 검색어가 '맛집', '카페' 등 일반 명사뿐이라면, **여행 목적지("{destination}")**를 정규화해서 반환하세요.
3. **절대 추측하지 마세요.** 모르면 "{destination}"을 그대로 반환하세요.
4. 답변에는 군더더기 없이 **오직 지역명만** 출력하세요.

[예시]
- 입력: "해운대 맛집", 목적지: "부산" -> 출력: "부산광역시 해운대구"
- 입력: "성산일출봉", 목적지: "제주도" -> 출력: "제주특별자치도 서귀포시"
- 입력: "강남 점심", 목적지: "서울" -> 출력: "서울특별시 강남구"
- 입력: "맛집 추천", 목적지: "여수" -> 출력: "전라남도 여수시"
""")
region_chain = region_prompt | LLM | StrOutputParser()

desc_prompt = PromptTemplate.from_template("""
[상황]
사용자 정보: {user_info}
장소 이름: {place_name}
장소 특징/리뷰 요약: {place_data}

위 정보를 바탕으로, 이 장소가 **이 사용자에게 왜 좋은지** 매력적인 1~2줄의 추천사를 작성해줘.
- 반드시 한국어로 작성.
- 문장 끝은 '해요', '좋아요' 등으로 자연스럽게 마무리.
""")
desc_chain = desc_prompt | LLM | StrOutputParser()
