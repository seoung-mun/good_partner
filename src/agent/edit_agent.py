"""일정 수정 에이전트: create_agent(tool calling) → 시간표 재계산 → 렌더. 무엇을 할지 예측할 수 없는 요청이라 에이전트로."""
import json

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware, ToolErrorMiddleware
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from src.agent.chains import format_plan_for_briefing
from src.agent.edit_tools import make_edit_tools
from src.agent.render import render_markdown
from src.agent.workflow import reschedule
from src.domain.models import ItineraryItem

SYSTEM = ("당신은 여행 일정 편집자입니다. 사용자의 수정 요청을 replace_slot·remove_slot·add_slot 도구로 반영하세요. "
          "슬롯은 반드시 [현재 일정]에 있는 slot_id로 지정하세요. 도구 결과의 ok가 false면 이유를 사용자에게 알리세요. "
          "요청을 반영했으면 무엇을 바꿨는지 한두 문장으로 답하고 끝내세요.")


def _tool_error(error: Exception, request) -> str:
    """tool 예외를 모델이 읽을 수 있는 실패 envelope로 (tool 정상 실패와 같은 형식)."""
    name = request.tool_call.get("name", "?") if hasattr(request, "tool_call") else "?"
    return json.dumps({"ok": False, "tool_name": name, "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False)


class EditState(AgentState):
    """수정 에이전트 상태: 대화 메시지 + 일정·요청·지역 + 사용자가 뺀 장소(재추천 금지)."""
    items: list[ItineraryItem]
    rejected: list[tuple[str, str]]
    request: dict
    region: str
    final_text: str


def build_edit_app(llm=None, load_db=None, route_fn=None):
    """수정 그래프: 에이전트(tool calling) → 시간표 재계산·렌더. 의존성 주입 가능."""
    if llm is None:
        from src.config import LLM as llm
    if load_db is None:
        from src.retrieval.store import load_faiss_index as load_db
    if route_fn is None:
        from src.services.maps import get_detailed_route as route_fn
    agent = create_agent(
        model=llm, tools=make_edit_tools(load_db), system_prompt=SYSTEM, state_schema=EditState,
        # 기존 코드의 재귀 감지·[SYSTEM ALERT] 주입을 대체하는 호출 상한과 tool 예외 처리
        middleware=[ModelCallLimitMiddleware(run_limit=6), ToolCallLimitMiddleware(run_limit=6), ToolErrorMiddleware(on_error=_tool_error)],
    )

    async def finalize(state: EditState) -> dict:
        """수정 반영 후 시간표 재계산·렌더. 경로 실패는 기본 이동 시간 + 안내."""
        items, errors = await reschedule(state["items"], route_fn, state["request"])
        return {"items": items, "final_text": render_markdown(state["request"], items, None, "", errors)}

    g = StateGraph(EditState)
    g.add_node("agent", agent)
    g.add_node("finalize", finalize)
    g.add_edge(START, "agent")
    g.add_edge("agent", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


async def edit(app, state: dict, user_text: str, config: dict | None = None) -> dict:
    """현재 일정 상태 + 사용자 요청 → 수정된 상태. 현재 일정은 slot_id와 함께 매 요청에 제공."""
    msg = HumanMessage(content=f"[현재 일정]\n{format_plan_for_briefing(state['items'])}\n\n[요청]\n{user_text}")
    return await app.ainvoke({**state, "messages": [msg]}, config=config)
