"""Task-specific heuristic allocation policy using theory-of-mind features."""
from __future__ import annotations

from models import StratArenaObservation


def _tom_dict(obs: StratArenaObservation) -> dict[str, float]:
    """Map the flat tom_features list to a named dict for readability."""
    features = obs.tom_features + ([0.0] * max(0, 10 - len(obs.tom_features)))
    return {
        "agg_budget": features[0],
        "agg_aggr": features[1],
        "agg_conf": features[2],
        "agg_vol": features[3],
        "con_budget": features[4],
        "con_aggr": features[5],
        "con_conf": features[6],
        "con_vol": features[7],
        "exploit": features[8],
        "uncertainty": features[9],
    }


def heuristic_allocation(obs: StratArenaObservation) -> tuple[float, str]:
    """Return (allocation, note) using hand-crafted ToM-aware heuristics.

    allocation is in [0.0, 2.0]:
      0   = skip this round
      1   = match estimated fair value
      2   = aggressive over-bid
    """
    tom = _tom_dict(obs)
    value_score = obs.resource_value * obs.market_pressure
    exploit_window = tom["exploit"] > 0.32 or tom["agg_budget"] < 0.45
    strong_pressure = obs.market_pressure > 1.35
    uncertain = obs.uncertainty_signal > 0.58 or tom["uncertainty"] > 0.58

    if obs.task == "easy":
        if value_score < 28.0:
            return 0.0, "skip weak"
        if value_score > 82.0 and obs.my_budget_ratio > 0.20:
            return 1.10, "take value"
        if exploit_window and value_score > 48.0:
            return 1.20, "press edge"
        return (0.65, "balanced probe") if value_score > 42.0 else (0.0, "wait")

    if obs.task == "medium":
        if exploit_window and value_score > 46.0:
            return 1.35, "exploit weak"
        if strong_pressure and not exploit_window:
            return 0.0, "avoid crowd"
        if uncertain and value_score < 60.0:
            return 0.0, "uncertain skip"
        if value_score > 72.0:
            return 1.00, "good value"
        return (0.55, "light entry") if value_score > 44.0 else (0.0, "pass")

    # hard task
    late_game = obs.step >= (obs.max_steps // 2)
    if late_game and exploit_window and value_score > 50.0:
        return 1.25, "adapt strike"
    if strong_pressure and value_score < 62.0:
        return 0.0, "protect budget"
    if uncertain and value_score < 68.0:
        return 0.0, "reassess"
    if value_score > 78.0:
        return 1.10, "high value"
    return (0.70, "selective test") if value_score > 54.0 else (0.0, "hold")
