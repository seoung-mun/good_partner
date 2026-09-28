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
