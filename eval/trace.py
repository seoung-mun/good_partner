"""에이전트 실행 추적: LLM·tool 호출 시간, tool 오류율, 중복 호출. 서비스 지표(지연 분해·오류율) 산출용."""
import math
import time
from collections import Counter

from langchain_core.callbacks import AsyncCallbackHandler


def percentile(xs: list[float], p: float) -> float | None:
    """nearest-rank 백분위수."""
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def timed(fn, acc: dict):
    """async 함수를 감싸 누적 시간(acc['t'])과 호출 수(acc['n'])를 기록 — 로직은 그대로."""
    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return await fn(*args, **kwargs)
        finally:
            acc["t"] += time.perf_counter() - t0
            acc["n"] += 1
    return wrapper


# 예외 없이 '실패'를 문자열로 돌려주는 legacy tool 응답 패턴
SOFT_ERROR_MARKERS = ("추천 장소 없음", "시스템에러", "최적화 실패", "오류:", "Error:", "API 키 없음")


class AgentTracer(AsyncCallbackHandler):
    """그래프 실행 config의 callbacks로 넘기면 하위 LLM·tool 이벤트까지 전파돼 수집된다."""

    def __init__(self, known_tools: set[str] | None = None):
        self.known_tools = known_tools
        self._start: dict = {}
        self.llm_durations: list[float] = []
        self.tool_events: list[dict] = []
        self.requested = 0
        self.invalid = 0

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        self._start[run_id] = time.perf_counter()

    async def on_llm_end(self, response, *, run_id, **kw):
        if run_id in self._start:
            self.llm_durations.append(time.perf_counter() - self._start.pop(run_id))
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                calls = getattr(msg, "tool_calls", None) or []
                bad = getattr(msg, "invalid_tool_calls", None) or []
                self.requested += len(calls) + len(bad)
                self.invalid += len(bad)
                if self.known_tools is not None:
                    self.invalid += sum(c["name"] not in self.known_tools for c in calls)

    async def on_llm_error(self, error, *, run_id, **kw):
        self._start.pop(run_id, None)

    async def on_tool_start(self, serialized, input_str, *, run_id, **kw):
        self._start[run_id] = (time.perf_counter(), (serialized or {}).get("name", "?"), str(input_str))

    async def _tool_done(self, run_id, error: bool, soft: bool = False):
        if run_id in self._start:
            t0, name, inp = self._start.pop(run_id)
            self.tool_events.append({"name": name, "input": inp, "error": error, "soft_error": soft,
                                     "duration": time.perf_counter() - t0})

    async def on_tool_end(self, output, *, run_id, **kw):
        text = str(getattr(output, "content", output))
        await self._tool_done(run_id, error=False, soft=any(m in text for m in SOFT_ERROR_MARKERS))

    async def on_tool_error(self, error, *, run_id, **kw):
        await self._tool_done(run_id, error=True)

    def summary(self) -> dict:
        ev = self.tool_events
        keys = Counter((e["name"], e["input"]) for e in ev)
        by_name_time = Counter()
        for e in ev:
            by_name_time[e["name"]] += e["duration"]
        return {
            "llm_calls": len(self.llm_durations),
            "llm_time_s": sum(self.llm_durations),
            "llm_p50_s": percentile(self.llm_durations, 50),
            "tool_calls": len(ev),
            "tool_errors": sum(e["error"] for e in ev),
            "tool_error_rate": (sum(e["error"] for e in ev) / len(ev)) if ev else None,
            "tool_soft_errors": sum(e["soft_error"] for e in ev),
            "tool_calls_requested": self.requested,
            "invalid_tool_calls": self.invalid,
            "invalid_tool_call_rate": (self.invalid / self.requested) if self.requested else None,
            "redundant_tool_calls": sum(c - 1 for c in keys.values() if c > 1),
            "tool_calls_by_name": dict(Counter(e["name"] for e in ev)),
            "tool_time_by_name_s": {k: round(v, 3) for k, v in by_name_time.items()},
        }
