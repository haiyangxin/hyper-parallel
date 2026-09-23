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
"""Exact resume checks must reject fresh optimizer state and altered saved moments."""

from copy import deepcopy

import pytest
import torch

from _repository_resume import assert_state_equal, optimizer_steps, yielded_cursors
from tests.common.mark_utils import arg_mark


_MARK = arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
                 card_mark="onecard", essential_mark="essential")


@_MARK
def test_restored_optimizer_compares_moments_not_only_step_counter() -> None:
    """A matching step cannot conceal failed moment restoration."""
    expected = {"state": {0: {"exp_avg": torch.tensor([0., 2.])}},
                "param_groups": [{"lr": 1e-6, "params": [0], "step": 30}]}
    actual = deepcopy(expected)
    assert_state_equal(actual, expected)
    assert optimizer_steps(actual) == {30}
    actual["state"][0]["exp_avg"][1] = 0
    with pytest.raises(RuntimeError, match="restored tensor differs"):
        assert_state_equal(actual, expected)
    for groups in ([], [{"params": [0]}]):
        with pytest.raises(RuntimeError, match="missing"):
            optimizer_steps({"param_groups": groups})
    with pytest.raises(RuntimeError, match="disagree"):
        optimizer_steps({"param_groups": [{"step": 30}, {"step": 0}]})


@_MARK
def test_restore_catches_poisoned_model_rng_and_loader_cursor() -> None:
    """Exact equality covers live parameters, RNG bytes and exhausted-loader progress."""
    expected = {"model": torch.tensor([1.]), "rng": torch.tensor([1, 2], dtype=torch.uint8),
                "loader": {"iterator": {"_num_yielded": 1}}}
    for key in ("model", "rng", "loader"):
        actual = deepcopy(expected)
        if key == "loader":
            actual[key]["iterator"]["_num_yielded"] = 0
        else:
            actual[key][0] = 0
        with pytest.raises(RuntimeError, match="restored"):
            assert_state_equal(actual, expected)
    assert yielded_cursors(expected["loader"]) == {"iterator._num_yielded": 1}
