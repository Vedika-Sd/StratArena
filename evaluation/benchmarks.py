from __future__ import annotations

from server.evaluate_tasks import run_task
from server.tasks import TASKS


def benchmark_all_tasks() -> list[dict[str, float | str | int]]:
    rows = []
    for task in TASKS:
        result = run_task(task)
        rows.append(
            {
                "task": result.task,
                "score": result.score,
                "value": result.total_value,
                "spend": result.spend,
                "wins": result.wins,
                "exploit": result.exploit,
                "belief": result.belief,
                "adaptation": result.adaptation,
            }
        )
    return rows
