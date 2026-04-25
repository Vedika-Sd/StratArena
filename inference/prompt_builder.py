"""Prompt construction utilities for StratArena LLM policies."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from models import StratArenaObservation


SYSTEM_PROMPT = """You are a strategic allocation policy for StratArena.
Return JSON only: {"allocation": <float 0-2>, "reason": "<short>"}
Use theory-of-mind signals:
- low opponent budget belief -> exploit with higher allocation on good value rounds
- high uncertainty or high market pressure -> reduce allocation unless value is strong
- hard task after the regime shift requires rapid adaptation, not constant aggression
Never output anything except JSON."""


@dataclass
class StepTrace:
    """Lightweight record of a single completed step for recent-history context."""

    step: int
    allocation: float
    reward: float
    exploit_signal: float
    task_score: float
    winner: str


def build_prompt(task: str, obs: StratArenaObservation, trace: list[StepTrace]) -> str:
    """Serialise the current observation and recent trace into a JSON string.

    The string is passed verbatim as the user message to the LLM policy.
    """
    recent_trace: list[dict[str, Any]] = []
    for trace_item in trace[-4:]:
        if hasattr(trace_item, "__dict__"):
            recent_trace.append(trace_item.__dict__)
        elif isinstance(trace_item, dict):
            recent_trace.append(trace_item)
        else:
            recent_trace.append({"value": str(trace_item)})

    payload: dict[str, Any] = {
        "task": task,
        "step": obs.step,
        "budget_ratio": round(obs.my_budget_ratio, 3),
        "resource_value": round(obs.resource_value, 2),
        "resource_scarcity": round(obs.resource_scarcity, 3),
        "market_pressure": round(obs.market_pressure, 3),
        "exploit_signal": round(obs.exploit_signal, 3),
        "uncertainty_signal": round(obs.uncertainty_signal, 3),
        "opponent_signals": obs.opponent_signals,
        "tom_features": obs.tom_features,
        "recent": recent_trace,
    }
    return json.dumps(payload)
