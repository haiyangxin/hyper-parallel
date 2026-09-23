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
"""Repository functional acceptance must not claim unobserved model learning."""

from copy import deepcopy

import pytest

from _repository_train import summarize_acceptance
from tests.common.mark_utils import arg_mark


_MARK = arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
                 card_mark="onecard", essential_mark="essential")


def _records() -> list[dict]:
    """Represent two valid steps with unequal calls, independent grading and zero reward."""
    records = []
    for step in (1, 2):
        ranks = []
        for rank, counts in enumerate(([2, 1], [2, 2])):
            ranks.append({"rank": rank, "dp_rank": rank, "physical_rows": 4,
                          "real_rows": sum(counts), "padding_rows": 4 - sum(counts),
                          "calls_per_episode": counts, "rewards": [0., 0.],
                          "sampled_version": step - 1, "published_version": step,
                          "optimizer_steps": 1, "valid_tokens": 10,
                          "gradient_norm": 0., "loss": 0., "parameter_max_delta": 0.,
                          "task_advantage_max_abs": 0., "task_advantage_nonzero_tokens": 0,
                          "mixed_reward_groups": [],
                          "episodes": [{"episode_id": f"{rank}:{step}:{index}",
                                        "reward": 0., "call_count": count,
                                        "artifact": "graded.manifest.json", "evaluated": 5}
                                       for index, count in enumerate(counts)]})
        records.append({"step": step, "ranks": ranks})
    return records


@_MARK
def test_functional_zero_rewards_do_not_claim_learning_or_repair() -> None:
    """Two real optimizer steps may be functional even when task advantages are zero."""
    records = _records()
    summary = summarize_acceptance(records, "functional", require_uneven=True)
    assert summary["functional_flow"] == "passed"
    assert summary["task_learning"] == "not_observed"
    assert summary["autonomous_repair"] == "not_observed"
    assert summary["graded_episodes"] == 8
    with pytest.raises(RuntimeError, match="Learning acceptance"):
        summarize_acceptance(records, "learning", require_uneven=True)


@_MARK
def test_learning_requires_coincident_task_signal_and_parameter_change() -> None:
    """Only genuine mixed task rewards with an update satisfy the strict mode."""
    records = _records()
    row = records[0]["ranks"][0]
    row.update(mixed_reward_groups=["task"], task_advantage_nonzero_tokens=5,
               task_advantage_max_abs=1., gradient_norm=0.1, parameter_max_delta=0.01)
    row["rewards"][0] = 1.
    row["episodes"][0]["reward"] = 1.
    assert summarize_acceptance(records, "learning")["task_learning"] == "passed"
    no_update = deepcopy(records)
    no_update[0]["ranks"][0]["parameter_max_delta"] = 0.
    with pytest.raises(RuntimeError, match="Learning acceptance"):
        summarize_acceptance(no_update, "learning")


@_MARK
@pytest.mark.parametrize("override", [
    {"optimizer_steps": 0}, {"valid_tokens": 0}, {"gradient_norm": float("nan")},
    {"published_version": 9}, {"padding_rows": 0},
])
def test_functional_retains_execution_and_alignment_gates(override: dict) -> None:
    """Functional mode still rejects skipped optimization, invalid numerics and misalignment."""
    records = _records()
    records[1]["ranks"][0].update(override)
    with pytest.raises(RuntimeError):
        summarize_acceptance(records, "functional")


@_MARK
def test_functional_requires_two_steps_and_real_independent_grading() -> None:
    """Model format failures alone cannot stand in for the repository workflow."""
    records = _records()
    with pytest.raises(RuntimeError, match="two completed"):
        summarize_acceptance(records[:1], "functional")
    for step in records:
        for row in step["ranks"]:
            for episode in row["episodes"]:
                episode["artifact"] = None
                episode["evaluated"] = 0
    with pytest.raises(RuntimeError, match="independent grader"):
        summarize_acceptance(records, "functional")
