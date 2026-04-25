"""StratArena inference package — public API.

Import from this package directly:

    from inference import heuristic_allocation, build_prompt, run_episode
"""
from __future__ import annotations

from .heuristic import heuristic_allocation
from .policy import (
    API_BASE_URL,
    API_KEY,
    MODEL_NAME,
    OpenAIPolicy,
    clamp_allocation,
    parse_json,
)
from .prompt_builder import SYSTEM_PROMPT, StepTrace, build_prompt
from .runner import (
    BENCHMARK,
    SUCCESS_SCORE_THRESHOLD,
    log_end,
    log_start,
    log_step,
    main,
    run_episode,
)

__all__ = [
    # heuristic
    "heuristic_allocation",
    # prompt builder
    "build_prompt",
    "SYSTEM_PROMPT",
    "StepTrace",
    # policy
    "OpenAIPolicy",
    "API_BASE_URL",
    "API_KEY",
    "MODEL_NAME",
    "clamp_allocation",
    "parse_json",
    # runner
    "BENCHMARK",
    "SUCCESS_SCORE_THRESHOLD",
    "log_start",
    "log_step",
    "log_end",
    "run_episode",
    "main",
]
