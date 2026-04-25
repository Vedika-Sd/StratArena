from __future__ import annotations

"""
Colab-oriented StratArena training entrypoint.

Recommended workflow:
1. Export rollouts locally or in Colab:
       python training/data_export.py --policy mixed --episodes-per-task 80
2. Upload the JSONL to Colab, or mount Drive.
3. Run SFT first to warm-start the model on observation -> action JSON.
4. Optionally continue with GRPO / RL using the reward functions below.

This script keeps imports lazy so the main repo does not require Unsloth locally.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are the learner policy for StratArena.
You receive a partial-observability game state.
Return only JSON with this schema:
{"allocation": <float 0-2>, "reason": "<short phrase>"}
Pick stronger allocations only when value is strong or an opponent looks weak.
Reduce allocation when pressure or uncertainty is high."""


def load_rollout_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_sft_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["prompt"]},
                ],
                "completion": row["completion"],
                "task": row["task"],
                "reward": row["reward"],
                "task_score": row["task_score"],
            }
        )
    return records


def build_grpo_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["prompt"]},
                ],
                "task": row["task"],
                "reference_reward": row["reward"],
                "reference_score": row["task_score"],
                "metadata": row.get("metadata", {}),
            }
        )
    return records


def _extract_allocation(text: str) -> float | None:
    if not text:
        return None
    match = re.search(r'"allocation"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return max(0.0, min(2.0, value))


def _parse_completion_text(completion_item: Any) -> str:
    if isinstance(completion_item, str):
        return completion_item
    if isinstance(completion_item, list):
        texts: list[str] = []
        for chunk in completion_item:
            if isinstance(chunk, dict):
                if chunk.get("type") == "text":
                    texts.append(str(chunk.get("text", "")))
                elif "content" in chunk:
                    texts.append(str(chunk["content"]))
            else:
                texts.append(str(chunk))
        return "".join(texts)
    if isinstance(completion_item, dict):
        return str(completion_item.get("content", completion_item))
    return str(completion_item)


def strat_arena_reward_func(prompts, completions, task, **kwargs):
    """
    GRPO reward function for StratArena.

    Each prompt already contains the serialized observation.
    The model's completion should emit {"allocation": ...}.
    We run one environment step and return the immediate reward plus a small
    shaping bonus from task score.
    """

    from models import StratArenaAction
    from server.stratarena_environment import StratArenaEnvironment

    rewards: list[float] = []
    for prompt_messages, completion, task_name in zip(prompts, completions, task):
        user_prompt = prompt_messages[-1]["content"]
        completion_text = _parse_completion_text(completion)
        allocation = _extract_allocation(completion_text)
        if allocation is None:
            rewards.append(-2.0)
            continue

        try:
            prompt_payload = json.loads(user_prompt)
        except json.JSONDecodeError:
            rewards.append(-1.5)
            continue

        env = StratArenaEnvironment()
        obs = env.reset(task=task_name, seed=17)
        target_step = int(prompt_payload.get("step", 0))
        for _ in range(target_step):
            if obs.done:
                break
            obs = env.step(StratArenaAction(allocation=0.0))

        obs = env.step(StratArenaAction(allocation=allocation))
        reward = float(obs.reward or 0.0) + (0.5 * float(obs.task_score))
        rewards.append(reward)
    return rewards


def format_reward_func(completions, **kwargs):
    rewards: list[float] = []
    for completion in completions:
        text = _parse_completion_text(completion)
        allocation = _extract_allocation(text)
        if allocation is None:
            rewards.append(-1.0)
            continue
        rewards.append(0.2)
    return rewards


def train_sft(
    dataset_path: str,
    model_name: str,
    output_dir: str,
    max_seq_length: int = 1024,
) -> None:
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    rows = load_rollout_rows(dataset_path)
    records = build_sft_records(rows)
    dataset = Dataset.from_list(records)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="completion",
        formatting_func=lambda batch: [
            tokenizer.apply_chat_template(sample["prompt"], tokenize=False) + sample["completion"]
            for sample in batch
        ],
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=100,
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def train_grpo(
    dataset_path: str,
    model_name: str,
    output_dir: str,
    max_prompt_length: int = 1024,
    max_completion_length: int = 96,
) -> None:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    rows = load_rollout_rows(dataset_path)
    records = build_grpo_records(rows)
    dataset = Dataset.from_list(records)

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[format_reward_func, strat_arena_reward_func],
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=output_dir,
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            max_steps=300,
            logging_steps=5,
            save_steps=50,
            report_to="none",
            bf16=True,
            use_vllm=False,
        ),
    )
    trainer.train()
    trainer.save_model(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="StratArena Unsloth / Colab training entrypoint")
    parser.add_argument("--dataset", required=True, help="Path to JSONL exported by training/data_export.py")
    parser.add_argument("--mode", choices=["sft", "grpo"], default="sft")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="outputs/stratarena_unsloth")
    args = parser.parse_args()

    if args.mode == "sft":
        train_sft(args.dataset, args.model, args.output_dir)
    else:
        train_grpo(args.dataset, args.model, args.output_dir)


if __name__ == "__main__":
    main()
