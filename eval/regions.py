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
