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
"""Explicit native Qwen3.5/Qwen3.8 text-policy RL acceptance worker."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

from rl.trainer import SyncTrainer


class Qwen3_5ObservedTrainer(SyncTrainer):
    """Observe the production trainer without altering actions, rewards or updates."""

    def __init__(self, config: dict, output: Path) -> None:
        """Bind persistent evidence before constructing the production runtime."""
        self.output = output
        self.records: list[dict[str, Any]] = []
        self.before: dict[str, torch.Tensor] = {}
        super().__init__(config)

    def _samples(self) -> dict[str, torch.Tensor]:
        """Observe finite local text parameters from both hybrid layer types."""
        samples = {}
        for name, parameter in self.actor.actor_model.named_parameters():
            if not name.startswith(("model.layers.", "model.embed_tokens.", "model.norm.", "lm_head.")):
                raise RuntimeError(f"Non-text Actor parameter {name!r}")
            if not parameter.requires_grad:
                continue
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            flat = local.detach().reshape(-1)
            if flat.numel():
                samples[name] = flat[::max(1, flat.numel() // 1024)][:1024].float().cpu().clone()
        for family in (".linear_attn.", ".self_attn."):
            if not any(family in name for name in samples):
                raise RuntimeError(f"Missing local hybrid parameter samples: {family}")
        if any(not torch.isfinite(value).all() for value in samples.values()):
            raise RuntimeError("Non-finite text policy samples")
        return samples

    def _train_step(self, batch: dict[str, Any]) -> None:
        self.before = self._run_rank_synchronized("text policy samples before update", self._samples)
        super()._train_step(batch)

    def _evidence(self, values: dict[str, Any]) -> dict[str, Any]:
        """Require finite optimization, real actions and current worker versions."""
        step, rollout, update = values["step"], values["rollout"], values["actor_update"]
        loss, norm = float(update.total_loss), float(update.gradient_norm)
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise RuntimeError("Non-finite policy loss or gradient norm")
        sampled_versions = {trajectory.policy_version for trajectory in rollout.trajectories}
        if sampled_versions != {step - 1} or rollout.worker_policy_version != step - 1:
            raise RuntimeError(f"Wrong sampled policy versions: {sampled_versions}")
        if self.rollout_engine.policy_version != step:
            raise RuntimeError("Weight publication did not commit the next policy version")
        valid = rollout.old_log_probs[rollout.loss_action_mask.bool()]
        if not valid.numel() or not torch.isfinite(valid).all():
            raise RuntimeError("No finite sampled action log probabilities")
        after = self._samples()
        if after.keys() != self.before.keys():
            raise RuntimeError("Text policy parameter identity changed during training")
        return {
            "rank": dist.get_rank(), "step": step, "loss": loss, "gradient_norm": norm,
            "sampled_version": step - 1, "published_version": self.rollout_engine.policy_version,
            "action_tokens": valid.numel(), "reward_sum": float(rollout.rewards.sum()),
            "parameter_sample_max_deltas": {
                name: float((after[name] - before).abs().max()) for name, before in self.before.items()
            },
        }

    def _complete_step(self, **values: Any) -> None:
        """Persist completed steps and require an actual update at final acceptance."""
        evidence = self._run_rank_synchronized("text policy step evidence", lambda: self._evidence(values))
        ranks: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, evidence)
        self.records.append({"step": values["step"], "ranks": ranks})

        def record() -> None:
            """Preserve evidence even when the final learning assertion fails."""
            if dist.get_rank() != 0:
                return
            changed = any(delta > 0 for row in self.records for rank in row["ranks"]
                          for delta in rank["parameter_sample_max_deltas"].values())
            self.output.mkdir(parents=True, exist_ok=True)
            result = {"config": self.resolved_config, "steps": self.records, "parameter_sample_changed": changed}
            (self.output / "training.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                      encoding="utf-8")
            if values["step"] == self.state.max_steps and (len(self.records) < 2 or not changed):
                raise RuntimeError("Acceptance requires two completed steps and a real parameter update")

        self._run_rank_synchronized("write text policy evidence", record)
        super()._complete_step(**values)


def main(argv: list[str] | None = None) -> None:
    """Run an explicit recipe with persistent version and parameter observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = Path(os.environ["RL_ST_RESULT_DIR"])
    trainer = Qwen3_5ObservedTrainer(yaml.safe_load(args.config.read_text(encoding="utf-8")), output)
    trainer.train()
    if int(os.environ.get("RANK", "0")) == 0:
        (output / "completed.json").write_text(json.dumps({"status": "passed", "steps": len(trainer.records)}) + "\n",
                                               encoding="utf-8")


def test_training() -> None:
    """Run native hybrid policy acceptance with the explicitly supplied resources."""
    main([os.environ["RL_ST_CONFIG"]])


if __name__ == "__main__":
    main()
