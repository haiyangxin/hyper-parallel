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
"""Observe only the requested one-step comparison without additional model forwards."""
import functools
import hashlib
import json
import os
import runpy
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from rl.roles.policy.actor import Actor
from rl.roles.rollout.vllm import VLLMGenerationEngine
from rl.trainer import SyncTrainer

ROOT = Path(os.environ["HYPER_COMPARISON_OUTPUT"])
RANK = int(os.environ["RANK"])
STATE = {"phase": None, "shapes": []}


def append(record: dict) -> None:
    """Persist each finished phase without relying on normal interpreter shutdown."""
    with (ROOT / f"phases-rank-{RANK}.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def measure(function: Callable, label: Optional[str], role: bool = False) -> Callable:
    """Time completed device work and record phase-local allocator peaks."""

    @functools.wraps(function)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Measure one phase and persist its completion before returning."""
        name = ("old_log_prob" if self.optimizer is not None else "reference") if role else label
        outer = STATE["phase"] is None
        initialized = torch.npu.is_initialized()
        if initialized:
            torch.npu.synchronize()
            if outer:
                torch.npu.reset_peak_memory_stats()
        previous = STATE["phase"]
        if outer:
            STATE.update(phase=name, shapes=[])
        start = time.time()
        record = {"phase": name, "started": start, "rank": RANK, "status": "failed"}
        try:
            result = function(self, *args, **kwargs)
            if torch.npu.is_initialized():
                torch.npu.synchronize()
            finish = time.time()
            record.update(
                finished=finish,
                seconds=finish - start,
                status="completed",
                allocated=torch.npu.memory_allocated(),
                reserved=torch.npu.memory_reserved(),
                peak_allocated=torch.npu.max_memory_allocated(),
                peak_reserved=torch.npu.max_memory_reserved(),
                peak_scope="phase" if outer else "parent",
            )
            if role:
                experience = args[0]
                shapes = list(STATE["shapes"])
                record.update(
                    shapes=shapes,
                    calls=len(shapes),
                    rows=experience.sequences.shape[0],
                    input_tokens=sum(x.token_ids.numel() for x in experience.trajectories),
                    padded_tokens=sum(rows * width for rows, width in shapes),
                )
                # These checks and host transfers are outside the measured device interval.
                values = result[experience.loss_action_mask.bool()]
                record.update(
                    valid_action_tokens=values.numel(),
                    nonfinite=int((~torch.isfinite(values)).sum().item()),
                    sequence_sha256=hashlib.sha256(experience.sequences.cpu().numpy().tobytes()).hexdigest(),
                )
                if record["nonfinite"]:
                    raise RuntimeError("Nonfinite training probabilities")
            return result
        finally:
            record.setdefault("finished", time.time())
            append(record)
            if outer:
                STATE.update(phase=previous, shapes=[])

    return wrapped


original_sequence = Actor.sequence_log_probs


@functools.wraps(original_sequence)
def sequence(self: Actor, sequences: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Record model input sizes without adding another forward."""
    if STATE["phase"] in ("old_log_prob", "reference"):
        STATE["shapes"].append(list(sequences.shape))
    return original_sequence(self, sequences, *args, **kwargs)


Actor.sequence_log_probs = sequence
Actor.compute_log_probs = measure(Actor.compute_log_probs, None, role=True)
Actor.update = measure(Actor.update, "actor_update")
SyncTrainer._publish_policy = measure(SyncTrainer._publish_policy, "weight_publish")
VLLMGenerationEngine._launch_client = measure(VLLMGenerationEngine._launch_client, "service_startup")
VLLMGenerationEngine.prepare_for_training = measure(VLLMGenerationEngine.prepare_for_training, "prepare_training")
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[4] / "train_rl.py"), run_name="__main__")
