import pytest
from langchain_core.documents import Document

from src.region_cut_fuzz import region_matches


@pytest.mark.parametrize("address,target,expected", [
    ("광주광역시 동구", "경기도 광주시", False),      # 광주광역시 ≠ 경기도 광주시
    ("경기도 광주시", "광주광역시", False),
    ("광주광역시 동구", "광주광역시", True),
    ("경기도 광주시", "경기 광주", True),             # 약칭 + 접미사 생략
    ("서울특별시 강서구", "부산광역시 강서구", False),  # 같은 구 이름, 다른 시도
    ("서울 강동구", "서울특별시 강동구", True),        # 메타데이터의 '서울' 표기 혼재
    ("강원특별자치도 고성군", "강원 고성", True),       # 고성 → 고성군 (시로 추정하지 않음)
    ("경상남도 고성군", "강원 고성", False),
    ("경기도 광주시", "경기도 광주시 퇴촌면", True),    # 읍면동까지 온 타깃은 시군구까지만 비교
    ("부산광역시 해운대구", "해운대구", True),          # 시도 없는 타깃은 시군구로 비교
    ("부산광역시 중구", "", True),                     # 타깃 없음 → 통과 (기존 동작)
    ("전북특별자치도 전주시", "전북도", True),          # LLM 정규화가 내놓는 비표준 약칭
    ("경상북도 경주시", "경북도", True),
    ("강원특별자치도 속초시", "강원도", True),
    ("경상남도 통영시", "경남도 통영시", True),
])
def test_region_matches(address, target, expected):
    assert region_matches(address, target) is expected


async def test_filter_candidates_excludes_other_gwangju():
    from src.tools import _filter_candidates
    docs = [Document(page_content="x", metadata={"장소명": "A", "지역": "광주광역시 서구", "카테고리": "식당"}),
            Document(page_content="x", metadata={"장소명": "B", "지역": "경기도 광주시", "카테고리": "식당"})]
    out = await _filter_candidates(docs, "경기도 광주시", [], "식당")
    assert [d.metadata["장소명"] for d in out] == ["B"]
