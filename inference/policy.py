"""OpenAI-backed allocation policy for StratArena."""
from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from .prompt_builder import SYSTEM_PROMPT, StepTrace, build_prompt
from models import StratArenaObservation


API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY: str | None = os.getenv("OPENAI_API_KEY")


def parse_json(raw: str) -> dict[str, Any]:
    """Parse a raw LLM response, stripping markdown code fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def clamp_allocation(value: Any) -> float:
    """Clamp any value to the valid allocation range [0.0, 2.0]."""
    try:
        return max(0.0, min(2.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


class OpenAIPolicy:
    """Allocation policy backed by an OpenAI-compatible chat completion API."""

    def __init__(self, model: str, api_key: str | None, base_url: str = API_BASE_URL) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def act(
        self,
        task: str,
        obs: StratArenaObservation,
        trace: list[StepTrace],
    ) -> tuple[float, str]:
        """Query the LLM and return (allocation, reason)."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(task, obs, trace)},
            ],
        )
        data = parse_json(response.choices[0].message.content or "{}")
        return clamp_allocation(data.get("allocation")), str(data.get("reason", "model"))
