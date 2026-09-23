# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Observe repository GRPO without changing rewards, actions or training updates."""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

from hyper_parallel.rl.tests.st._agent_train import AgentObservedTrainer, _validate_rank_alignment
from rl.dataset.episodes import episode_rows


class RepositoryObservedTrainer(AgentObservedTrainer):
    """Add task-learning and independently graded artifact evidence to agent checks."""

    def __init__(self, config: dict, output: Path) -> None:
        """Initialize evidence before constructing the normal trainer runtime."""
        self.task_evidence: dict = {}
        super().__init__(config, output)

    def _prepare_experience(self, rollout: Any, collect_diagnostics: bool,
                            timings: dict[str, float]) -> tuple[Any, dict[str, float]]:
        experience, metrics = super()._prepare_experience(rollout, collect_diagnostics, timings)

        def inspect_advantages() -> dict:
            """Record task advantages while rejecting any contribution from DP padding."""
            advantages = experience.advantages
            if advantages is None or not torch.isfinite(advantages).all():
                raise RuntimeError("Repository training requires finite task advantages")
            mask = experience.loss_action_mask.bool()
            padding = [index for index, row in enumerate(experience.trajectories)
                       if row.metadata.get("dp_padding", False)]
            if padding and (advantages[padding].any().item() or mask[padding].any().item()):
                raise RuntimeError("Repository padding contributes advantages or loss")
            selected = advantages[mask]
            return {"task_advantage_nonzero_tokens": int(torch.count_nonzero(selected).item()),
                    "task_advantage_max_abs": float(selected.abs().max()) if selected.numel() else 0.0}

        self.task_evidence = self._run_rank_synchronized("repository task advantages", inspect_advantages)
        return experience, metrics

    def _evidence(self, values: dict) -> dict:
        evidence = super()._evidence(values)
        rollout, update = values["rollout"], values["actor_update"]
        episodes, groups = [], {}
        for indices in episode_rows(rollout.trajectories):
            row = rollout.trajectories[indices[0]]
            metadata = row.metadata
            groups.setdefault(row.prompt_id, []).append(float(row.reward))
            item = {"episode_id": metadata["episode_id"], "prompt_id": row.prompt_id,
                    "reward": float(row.reward), "call_count": len(indices),
                    "status": metadata.get("status"), "artifact_dir": metadata.get("artifact_dir"),
                    "artifact": metadata.get("artifact"), "artifact_hash": metadata.get("artifact_hash"),
                    "evaluated": metadata.get("evaluated", 0),
                    "truncated": row.truncated, "terminal_reason": row.terminal_reason,
                    "budget_termination": metadata.get("budget_termination")}
            if item["artifact"]:
                manifest = json.loads(Path(item["artifact"]).read_text(encoding="utf-8"))
                if manifest["content_hash"] != item["artifact_hash"]:
                    raise RuntimeError("Repository artifact hash disagrees with grader evidence")
                item["changes"] = manifest["changes"]
            if row.reward == 1:
                if (item["status"] != "passed" or item["evaluated"] != 5
                        or not item.get("changes") or not any(item["changes"].values())):
                    raise RuntimeError("Repository success requires independently graded source changes")
            episodes.append(item)
        evidence.update(self.task_evidence)
        evidence.update({"episodes": episodes, "optimizer_steps": int(update.optimizer_steps),
                         "mixed_reward_groups": [key for key, rewards in groups.items() if len(set(rewards)) > 1]})
        if update.optimizer_steps <= 0:
            raise RuntimeError("Repository step did not execute the optimizer")
        return evidence


def summarize_acceptance(records: list[dict], acceptance: str, require_uneven: bool = False) -> dict:
    """Separate functional execution from observed model repair and task learning."""
    if acceptance not in {"functional", "learning"}:
        raise ValueError("Unknown repository acceptance mode")
    if len(records) < 2:
        raise RuntimeError("Repository acceptance requires two completed policy publications")
    rows = [row for step in records for row in step["ranks"]]
    for index, step in enumerate(records, start=1):
        if step["step"] != index or not step["ranks"]:
            raise RuntimeError("Repository acceptance requires consecutive observed steps from policy zero")
        for row in step["ranks"]:
            if row["sampled_version"] != index - 1 or row["published_version"] != index:
                raise RuntimeError("Repository acceptance requires sampling and publishing consecutive policies")
            if row["optimizer_steps"] <= 0 or row["valid_tokens"] <= 0:
                raise RuntimeError("Repository acceptance requires optimizer execution on real action tokens")
            if any(not math.isfinite(row[key]) for key in (
                    "gradient_norm", "loss", "parameter_max_delta", "task_advantage_max_abs")):
                raise RuntimeError("Repository acceptance requires finite update evidence")
    episodes = [episode for row in rows for episode in row["episodes"]]
    graded_episodes = {episode["episode_id"] for episode in episodes
                       if episode.get("artifact") and episode["evaluated"] == 5}
    if not any(episode["call_count"] > 1 and episode.get("artifact") and episode["evaluated"] == 5
               for episode in episodes):
        raise RuntimeError("No real multi-call repository submission reached the independent grader")
    learning = any(row["mixed_reward_groups"] and row["task_advantage_nonzero_tokens"] > 0
                   and row["gradient_norm"] > 0 and row["parameter_max_delta"] > 0 for row in rows)
    repair = any(episode["reward"] == 1 and episode["call_count"] > 1 for episode in episodes)
    alignment = [_validate_rank_alignment(step["ranks"]) for step in records]
    uneven = any(alignment)
    if require_uneven and not uneven:
        raise RuntimeError("Repository acceptance requires real unequal call counts and zero-loss DP padding")
    if acceptance == "learning" and not (learning and repair):
        raise RuntimeError("Learning acceptance requires autonomous repair and genuine task-driven parameter changes")
    return {"status": "passed", "acceptance": acceptance, "steps": len(records),
            "graded_episodes": len(graded_episodes),
            "uneven_calls_observed": uneven, "functional_flow": "passed",
            "task_learning": "passed" if learning else "not_observed",
            "autonomous_repair": "passed" if repair else "not_observed"}


def main() -> None:
    """Run via torchrun with explicit functional or strict learning acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--require-uneven-calls", action="store_true")
    parser.add_argument("--acceptance", choices=("functional", "learning"), default="learning")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if float(config["algorithm"].get("kl_coef", 0)) or float(config["train"]["optimizer"].get("weight_decay", 0)):
        raise ValueError("Repository learning acceptance isolates task reward: KL and weight decay must be zero")
    trainer = RepositoryObservedTrainer(config, args.output)
    rank = dist.get_rank()
    trainer.train()
    records = trainer.records
    summary = summarize_acceptance(records, args.acceptance, args.require_uneven_calls)
    final_step = records[-1]["step"]
    if trainer.evaluator is None or trainer.evaluator.last_step != final_step:
        raise RuntimeError("Repository acceptance requires evaluation under the final published policy")
    checkpoint = trainer.checkpoints.directory(final_step) / "checkpoint_complete.json"
    if not checkpoint.is_file():
        raise RuntimeError("Repository acceptance requires the final checkpoint completion manifest")
    checkpoint_metadata = json.loads(checkpoint.read_text(encoding="utf-8"))
    if checkpoint_metadata["step"] != final_step or checkpoint_metadata["world_size"] != len(records[-1]["ranks"]):
        raise RuntimeError("Final checkpoint manifest does not match the observed step and world size")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "completed.json").write_text(json.dumps({
            **summary,
            "evaluation_step": final_step, "checkpoint": str(checkpoint), "checkpoint_reload": "not_run",
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
