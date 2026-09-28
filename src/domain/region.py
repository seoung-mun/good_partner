"""행정구역 정규화·일치 판정·목적지 해석 (시/도 + 시/군/구)."""

REGION_STRUCTURE = {
    "서울특별시": ["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"],
    "부산광역시": ["중구", "서구", "동구", "영도구", "부산진구", "동래구", "남구", "북구", "해운대구", "사하구", "금정구", "강서구", "연제구", "수영구", "사상구", "기장군"],
    "대구광역시": ["중구", "동구", "서구", "남구", "북구", "수성구", "달서구", "달성군", "군위군"],
    "인천광역시": ["중구", "동구", "미추홀구", "연수구", "남동구", "부평구", "계양구", "서구", "강화군", "옹진군"],
    "광주광역시": ["동구", "서구", "남구", "북구", "광산구"],
    "대전광역시": ["동구", "중구", "서구", "유성구", "대덕구"],
    "울산광역시": ["중구", "남구", "동구", "북구", "울주군"],
    "세종특별자치시": [],
    "경기도": ["수원시", "성남시", "의정부시", "안양시", "부천시", "광명시", "평택시", "동두천시", "안산시", "고양시", "과천시", "구리시", "남양주시", "오산시", "시흥시", "군포시", "의왕시", "하남시", "용인시", "파주시", "이천시", "안성시", "김포시", "화성시", "광주시", "양주시", "포천시", "여주시", "연천군", "가평군", "양평군"],
    "강원특별자치도": ["춘천시", "원주시", "강릉시", "동해시", "태백시", "속초시", "삼척시", "홍천군", "횡성군", "영월군", "평창군", "정선군", "철원군", "화천군", "양구군", "인제군", "고성군", "양양군"],
    "충청북도": ["청주시", "충주시", "제천시", "보은군", "옥천군", "영동군", "증평군", "진천군", "괴산군", "음성군", "단양군"],
    "충청남도": ["천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시", "당진시", "금산군", "부여군", "서천군", "청양군", "홍성군", "예산군", "태안군"],
    "전북특별자치도": ["전주시", "군산시", "익산시", "정읍시", "남원시", "김제시", "완주군", "진안군", "무주군", "장수군", "임실군", "순창군", "고창군", "부안군"],
    "전라남도": ["목포시", "여수시", "순천시", "나주시", "광양시", "담양군", "곡성군", "구례군", "고흥군", "보성군", "화순군", "장흥군", "강진군", "해남군", "영암군", "무안군", "함평군", "영광군", "장성군", "완도군", "진도군", "신안군"],
    "경상북도": ["포항시", "경주시", "김천시", "안동시", "구미시", "영주시", "영천시", "상주시", "문경시", "경산시", "의성군", "청송군", "영양군", "영덕군", "청도군", "고령군", "성주군", "칠곡군", "예천군", "봉화군", "울진군", "울릉군"],
    "경상남도": ["창원시", "진주시", "통영시", "사천시", "김해시", "밀양시", "거제시", "양산시", "의령군", "함안군", "창녕군", "고성군", "남해군", "하동군", "산청군", "함양군", "거창군", "합천군"],
    "제주특별자치도": ["제주시", "서귀포시"]
}

SIMPLE_ALIASES = {
    "서울": "서울특별시", "서울시": "서울특별시",
    "부산": "부산광역시", "부산시": "부산광역시",
    "대구": "대구광역시", "대구시": "대구광역시",
    "인천": "인천광역시", "인천시": "인천광역시",
    "광주": "광주광역시", "광주시": "광주광역시",
    "대전": "대전광역시", "대전시": "대전광역시",
    "울산": "울산광역시", "울산시": "울산광역시",
    "세종": "세종특별자치시", "세종시": "세종특별자치시",
    "경기": "경기도",
    "강원": "강원특별자치도", "강원도": "강원특별자치도",
    "충북": "충청북도", "충청북": "충청북도",
    "충남": "충청남도", "충청남": "충청남도",
    "전북": "전북특별자치도", "전라북도": "전북특별자치도", "전북특별자치": "전북특별자치도",
    "전남": "전라남도", "전라남": "전라남도",
    "경북": "경상북도", "경상북": "경상북도",
    "경남": "경상남도", "경상남": "경상남도",
    "제주": "제주특별자치도", "제주도": "제주특별자치도", "제주특별자치": "제주특별자치도"
}

def normalize_region_name(user_input: str) -> str:
    """
    사용자가 입력한 목적지(예: '서울', '부산 해운대')를 
    공식 행정구역 명칭(예: '서울특별시', '부산광역시')으로 변환합니다.
    """
    if not user_input:
        return ""
    
    clean = user_input.strip()
    
    if clean in REGION_STRUCTURE:
        return clean
    if clean in SIMPLE_ALIASES:
        return SIMPLE_ALIASES[clean]
        

    parts = clean.split()
    if len(parts) >= 2:
        head = parts[0] 
        if head in REGION_STRUCTURE:
            return head
        if head in SIMPLE_ALIASES:
            return SIMPLE_ALIASES[head]
            

    return clean

def _sido_of(token: str) -> str:
    """시/도 토큰 → 정식 명칭. 약칭('부산'), 옛 명칭('강원도'), LLM식 표기('전북도')를 인식."""
    if token in REGION_STRUCTURE:
        return token
    # LLM이 '전북도', '경북도'처럼 약칭 + '도'로 내놓는 경우도 인식
    return SIMPLE_ALIASES.get(token) or SIMPLE_ALIASES.get(token.removesuffix("도"), "")


def _sigungu_of(token: str, sido: str) -> str:
    """'광주' → '광주시'(경기도), '고성' → '고성군'(강원). 행정구역 표에서 접미사까지 복원."""
    for name in REGION_STRUCTURE.get(sido, []):
        if token == name or token == name[:-1]:
            return name
    return token


def canonical_region(text: str) -> tuple[str, str]:
    """'서울 강동구' → ('서울특별시', '강동구'). 시도를 못 찾으면 ('', 첫 토큰)."""
    parts = (text or "").split()
    if not parts:
        return ("", "")
    sido = _sido_of(parts[0])
    if not sido:
        return ("", parts[0])
    return (sido, _sigungu_of(parts[1], sido) if len(parts) > 1 else "")


def region_matches(address: str, target: str) -> bool:
    """후보 주소가 목표 지역(시도 [+ 시군구]) 안에 있는지. 동명 시군구(중구·강서구·고성군, 광주)를 구분한다."""
    t_sido, t_sigungu = canonical_region(target)
    if not t_sido and not t_sigungu:
        return True
    a_sido, a_sigungu = canonical_region(address)
    if not t_sido:  # 시도 없는 타깃('해운대구')은 시군구만 비교
        return a_sigungu == t_sigungu
    return a_sido == t_sido and (not t_sigungu or a_sigungu == t_sigungu)


def resolve_destination(text: str) -> tuple[str, str] | None:
    """사용자 목적지 → (시도, 시군구). 시군구만 적었으면 전국에서 유일할 때만 확정, 모호하면 None."""
    parts = (text or "").split()
    if not parts:
        return None
    sido = _sido_of(parts[0])
    if sido:
        if len(parts) == 1:
            return (sido, "")
        sigungu = _sigungu_of(parts[1], sido)
        return (sido, sigungu if sigungu in REGION_STRUCTURE[sido] else "")
    hits = [(s, g) for s, names in REGION_STRUCTURE.items() for g in names if parts[0] in (g, g[:-1])]
    return hits[0] if len(hits) == 1 else None

