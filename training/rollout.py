from __future__ import annotations

from typing import Callable

from models import StratArenaAction, StratArenaObservation
from server.stratarena_environment import StratArenaEnvironment
from training.metrics import EpisodeMetrics


PolicyFn = Callable[[StratArenaObservation], float]


def run_rollout(task: str, policy: PolicyFn, seed: int | None = None) -> EpisodeMetrics:
    env = StratArenaEnvironment()
    obs = env.reset(task=task, seed=seed)
    while not obs.done:
        allocation = max(0.0, min(2.0, float(policy(obs))))
        obs = env.step(StratArenaAction(allocation=allocation))

    summary = env.summary_metrics()
    return EpisodeMetrics(
        task=task,
        score=env.grade(),
        total_value=obs.total_value_won,
        spend=obs.spend_so_far,
        wins=obs.wins,
        exploit=summary["exploit_success_rate"],
        belief=summary["belief_alignment"],
        adaptation=summary["adaptation_score"],
    )
