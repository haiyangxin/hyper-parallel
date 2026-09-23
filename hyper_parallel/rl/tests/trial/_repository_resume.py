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
"""Verify actual step-two restoration before one repository continuation step."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

from hyper_parallel.rl.tests.st._agent_train import _validate_rank_alignment
from _repository_train import RepositoryObservedTrainer
from hyper_parallel.core.distributed_checkpoint import load as dcp_load


def _local(value: Any) -> Any:
    """Compare local DTensor shards without triggering redistribution."""
    return value.to_local() if hasattr(value, "to_local") else value


def assert_state_equal(actual: Any, expected: Any, path: str = "state") -> None:
    """Require exact restored values, including optimizer moments and RNG byte tensors."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise RuntimeError(f"{path}: restored value is not a tensor")
        left, right = _local(actual).detach().cpu(), _local(expected).detach().cpu()
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise RuntimeError(f"{path}: restored tensor differs from checkpoint")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise RuntimeError(f"{path}: restored mapping keys differ")
        for key, value in expected.items():
            assert_state_equal(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise RuntimeError(f"{path}: restored sequence differs")
        for index, value in enumerate(expected):
            assert_state_equal(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not np.array_equal(actual, expected):
            raise RuntimeError(f"{path}: restored array differs")
    elif actual != expected:
        raise RuntimeError(f"{path}: restored scalar differs")


def optimizer_steps(state: dict) -> set[int]:
    """Require matching HyperAdamW parameter-group step counters."""
    groups = state.get("param_groups", [])
    if not groups or any("step" not in group for group in groups):
        raise RuntimeError("Restored HyperAdamW is missing parameter-group step counters")
    values = {int(_local(group["step"]).item()) if torch.is_tensor(group["step"]) else int(group["step"])
              for group in groups}
    if len(values) != 1:
        raise RuntimeError("Restored HyperAdamW parameter-group step counters disagree")
    return values


def yielded_cursors(state: dict) -> dict:
    """Keep the loader's yielded counters readable without treating the next prompt as proof."""
    result = {}
    for key, value in state.items():
        if "yielded" in str(key):
            result[str(key)] = value
        if isinstance(value, dict):
            result.update({f"{key}.{nested}": count for nested, count in yielded_cursors(value).items()})
    return result


class ResumedRepositoryTrainer(RepositoryObservedTrainer):
    """Wrap only the production resume boundary, before any policy publication."""

    def __init__(self, config: dict, output: Path) -> None:
        """Bind observation after construction without replacing production restore code."""
        super().__init__(config, output)
        self.resume_evidence: dict = {}
        self._production_begin = self.checkpoints.begin
        self.checkpoints.begin = self._checked_begin

    def _checked_begin(self, state: Any) -> None:
        """Poison live Actor state, run real restoration, and verify before publication."""
        checkpoint = Path(self.checkpoints.load_path)
        expected_model = self._run_rank_synchronized("resume expected model allocation", lambda: {
            name: value.detach().clone() for name, value in self.model.state_dict().items()
        })
        model_state = {"model": expected_model}
        self._run_rank_synchronized("resume expected model", lambda: dcp_load(
            model_state, checkpoint_id=checkpoint, use_collectives=True,
        ))
        runtime: dict[str, Any] = {"runtime": b""}
        self._run_rank_synchronized("resume expected runtime", lambda: dcp_load(
            runtime, checkpoint_id=checkpoint / f"rank_{dist.get_rank()}", use_collectives=False,
        ))
        def read_progress() -> dict:
            """Reject inconsistent rank-local input before another distributed operation."""
            if not isinstance(runtime["runtime"], dict):
                raise RuntimeError("Checkpoint runtime did not deserialize to a mapping")
            rank_state = dict(runtime["runtime"])
            progress = json.loads((checkpoint / "extra_state.json").read_text())
            if progress["global_step"] != 2 or optimizer_steps(rank_state["optimizer"]) != {30}:
                raise RuntimeError("Resume acceptance requires the real step-two / optimizer-thirty checkpoint")
            return progress

        progress = self._run_rank_synchronized("resume expected progress", read_progress)
        expected = dict(runtime["runtime"])

        def poison() -> dict:
            """Change one live local parameter so skipped restoration is observable."""
            name, parameter = next((name, value) for name, value in self.actor.actor_model.named_parameters()
                                   if _local(value).numel())
            element = _local(parameter).view(-1)[0]
            before = float(element.detach().cpu())
            with torch.no_grad():
                element.fill_(123.0 if before != 123.0 else -123.0)
            return {"name": name, "before": before, "poisoned": float(element.detach().cpu())}

        changed = self._run_rank_synchronized("resume poison live Actor", poison)
        self._production_begin(state)

        def verify() -> dict:
            """Check all restored components before trainer.train can publish weights."""
            assert_state_equal(self.model.state_dict(), model_state["model"], "model")
            actual_optimizer = self.checkpoints._optimizer_state_dict()
            assert_state_equal(actual_optimizer, expected["optimizer"], "optimizer")
            assert_state_equal(self.lr_scheduler.state_dict(), expected["scheduler"], "scheduler")
            assert_state_equal(torch.get_rng_state(), expected["cpu_rng"], "cpu_rng")
            assert_state_equal(self.device_handle.get_rng_state(self.device), expected["device_rng"], "device_rng")
            # StatefulDataLoader.state_dict lazily creates an iterator and consumes RNG.
            # Inspect the exact state installed by load_state_dict without changing that lifecycle.
            if self.train_dataloader._iterator is not None:
                raise RuntimeError("Restored dataloader was iterated before resume verification")
            assert_state_equal(self.train_dataloader.next_iter_state, expected["dataloader"], "dataloader")
            for key, value in progress.items():
                if getattr(state, key) != value:
                    raise RuntimeError(f"Restored progress mismatch: {key}")
            if self.rollout_engine.policy_version != 0:
                raise RuntimeError("Resume verification ran after a policy was already published")
            assert_state_equal(torch.get_rng_state(), expected["cpu_rng"], "cpu_rng_after_observation")
            assert_state_equal(self.device_handle.get_rng_state(self.device), expected["device_rng"],
                               "device_rng_after_observation")
            return {"rank": dist.get_rank(), "global_step": state.global_step,
                    "restored_epoch": state.epoch, "dataloader_yielded": yielded_cursors(expected["dataloader"]),
                    "optimizer_steps": sorted(optimizer_steps(actual_optimizer)),
                    "scheduler": expected["scheduler"], "poison": changed,
                    "model_exact": True, "optimizer_exact": True, "rng_exact": True,
                    "dataloader_exact": True, "verified_before_publication": True}

        self.resume_evidence = self._run_rank_synchronized("resume exact restored state", verify)

        def save_evidence() -> None:
            """Persist rank evidence before any peer proceeds to policy publication."""
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / f"restored-rank-{dist.get_rank()}.json").write_text(
                json.dumps(self.resume_evidence, indent=2, allow_nan=False) + "\n",
            )

        self._run_rank_synchronized("resume evidence write", save_evidence)

    def _train_step(self, batch: dict) -> None:
        """Require the restored policy to be served before genuine continuation sampling."""
        def validate() -> None:
            """Reject one rank's invalid restoration before any peer samples or trains."""
            if self.state.global_step != 2 or self.rollout_engine.policy_version != 2 or not self.resume_evidence:
                raise RuntimeError("Continuation must start from the verified restored policy two")
            if self.state.epoch != self.resume_evidence["restored_epoch"] + 1:
                raise RuntimeError("The exhausted restored loader did not advance to its next epoch")

        self._run_rank_synchronized("resume continuation preflight", validate)
        super()._train_step(batch)

    def _evidence(self, values: dict) -> dict:
        """Inspect optimizer progress while the process group is still live."""
        evidence = super()._evidence(values)
        counts = optimizer_steps(self.checkpoints._optimizer_state_dict())
        if counts != {30 + evidence["optimizer_steps"]}:
            raise RuntimeError("Resumed optimizer counters did not advance from thirty")
        evidence["restored_optimizer_steps"] = 30
        evidence["optimizer_steps_after"] = next(iter(counts))
        evidence["epoch_after"] = self.state.epoch
        return evidence


def main() -> None:
    """Continue the production trainer for exactly one step after exact restore checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text())
    if config["train"]["max_steps"] != 3 or not config["train"]["checkpoint"].get("load_path"):
        raise ValueError("Resume acceptance requires max_steps=3 and an explicit step-two checkpoint")
    trainer = ResumedRepositoryTrainer(config, args.output)
    rank = dist.get_rank()
    trainer.train()
    if len(trainer.records) != 1 or trainer.records[0]["step"] != 3:
        raise RuntimeError("Resume acceptance requires exactly one completed step-three publication")
    _validate_rank_alignment(trainer.records[0]["ranks"])
    for row in trainer.records[0]["ranks"]:
        if row["sampled_version"] != 2 or row["published_version"] != 3 or row["optimizer_steps"] <= 0:
            raise RuntimeError("Resumed sampling/publication/optimizer evidence is invalid")
    expected_count = trainer.records[0]["ranks"][rank]["optimizer_steps_after"]
    checkpoint = trainer.checkpoints.directory(3)
    if (trainer.evaluator is None or trainer.evaluator.last_step != 3
            or not (checkpoint / "checkpoint_complete.json").is_file() or not (checkpoint / "hf").is_dir()):
        raise RuntimeError("Resume acceptance requires final evaluation, checkpoint and HF export")
    if rank == 0:
        (args.output / "completed.json").write_text(json.dumps({
            "status": "passed", "restored_step": 2, "continued_step": 3,
            "optimizer_steps_before": 30, "optimizer_steps_after": expected_count,
            "checkpoint": str(checkpoint), "evaluation_step": 3,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
