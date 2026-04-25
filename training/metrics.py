from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    task: str
    score: float
    total_value: float
    spend: float
    wins: int
    exploit: float
    belief: float
    adaptation: float


def format_metrics(metrics: EpisodeMetrics) -> str:
    return (
        f"task={metrics.task} score={metrics.score:.4f} value={metrics.total_value:.2f} "
        f"spend={metrics.spend:.2f} wins={metrics.wins} exploit={metrics.exploit:.3f} "
        f"belief={metrics.belief:.3f} adapt={metrics.adaptation:.3f}"
    )
