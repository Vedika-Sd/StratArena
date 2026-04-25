"""Episode runner and CLI entry point for StratArena inference."""
from __future__ import annotations

import argparse

from .heuristic import heuristic_allocation
from .policy import API_BASE_URL, API_KEY, MODEL_NAME, OpenAIPolicy
from .prompt_builder import StepTrace
from models import StratArenaAction
from server.stratarena_environment import StratArenaEnvironment


BENCHMARK = "stratarena"
SUCCESS_SCORE_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Structured logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    joined = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={joined}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

def run_episode(
    task: str,
    *,
    model: str = MODEL_NAME,
    seed: int | None = None,
    use_llm: bool = False,
    base_url: str = API_BASE_URL,
    api_key: str | None = None,
) -> float:
    """Run one episode and return the final task score.

    Args:
        task: One of ``"easy"``, ``"medium"``, ``"hard"``.
        model: Model name for the OpenAI API (ignored when use_llm=False).
        seed: RNG seed for reproducibility.
        use_llm: If True and an API key is available, use the LLM policy;
            otherwise fall back to the heuristic.
        base_url: Base URL for the OpenAI-compatible API.
        api_key: Override for the OPENAI_API_KEY environment variable.

    Returns:
        Task score in [0.0, 1.0].
    """
    env = StratArenaEnvironment()
    obs = env.reset(task=task, seed=seed)
    policy = (
        OpenAIPolicy(model=model, api_key=api_key or API_KEY, base_url=base_url)
        if use_llm and (api_key or API_KEY)
        else None
    )

    rewards: list[float] = []
    trace: list[StepTrace] = []
    steps_taken = 0

    log_start(task, BENCHMARK, model if policy else "heuristic")
    while not obs.done:
        error: str | None = None
        if policy is None:
            allocation, note = heuristic_allocation(obs)
        else:
            try:
                allocation, note = policy.act(task, obs, trace)
            except Exception as exc:
                allocation, note = heuristic_allocation(obs)
                error = str(exc)

        obs = env.step(StratArenaAction(allocation=allocation))
        reward = float(obs.reward or 0.0)
        rewards.append(reward)
        steps_taken += 1
        log_step(steps_taken, f"allocation={allocation:.3f}", reward, obs.done, error)
        trace.append(
            StepTrace(
                step=obs.step,
                allocation=allocation,
                reward=reward,
                exploit_signal=obs.exploit_signal,
                task_score=obs.task_score,
                winner=obs.last_winner,
            )
        )

    score = env.grade()
    log_end(score >= SUCCESS_SCORE_THRESHOLD, steps_taken, score, rewards)
    return score


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="StratArena inference runner")
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--policy", default="heuristic", choices=["heuristic", "openai"])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-url", default=API_BASE_URL)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    scores: dict[str, float] = {}
    for task in tasks:
        scores[task] = run_episode(
            task,
            model=args.model,
            seed=args.seed,
            use_llm=args.policy == "openai",
            base_url=args.base_url,
            api_key=args.api_key,
        )

    if len(scores) > 1:
        average = sum(scores.values()) / len(scores)
        print(
            "[DEBUG] SUMMARY "
            + " | ".join(f"{t}={s:.3f}" for t, s in scores.items())
            + f" | avg={average:.3f}"
        )


if __name__ == "__main__":
    main()
