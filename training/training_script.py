from __future__ import annotations

import random

from inference import heuristic_allocation
from server.tasks import TASKS
from training.metrics import format_metrics
from training.rollout import run_rollout


def random_policy_factory(seed: int) -> callable:
    rng = random.Random(seed)
    return lambda obs: rng.uniform(0.0, 2.0)


def heuristic_policy(obs):
    allocation, _ = heuristic_allocation(obs)
    return allocation


def main() -> None:
    print("StratArena baseline rollouts")
    print("-" * 80)
    for idx, task in enumerate(TASKS, start=1):
        random_metrics = run_rollout(task, random_policy_factory(100 + idx), seed=100 + idx)
        heuristic_metrics = run_rollout(task, heuristic_policy, seed=200 + idx)
        print("random   " + format_metrics(random_metrics))
        print("heuristic " + format_metrics(heuristic_metrics))
        print("-" * 80)


if __name__ == "__main__":
    main()
