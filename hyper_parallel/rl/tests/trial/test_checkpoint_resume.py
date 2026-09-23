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
"""Reject checkpoint progress that would publish weights under the wrong version."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl.checkpoint import RLCheckpointManager
from tests.common.mark_utils import arg_mark


_MARK = arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
                 card_mark="onecard", essential_mark="essential")


@_MARK
@pytest.mark.parametrize("manifest_step,progress,valid", [
    (2, {"global_step": 2}, True),
    (2, {"global_step": 1}, False),
    (2, {}, False),
    (None, {"global_step": 2}, False),
    (True, {"global_step": 1}, False),
    (1, {"global_step": True}, False),
    (-1, {"global_step": -1}, False),
    ("2", {"global_step": "2"}, False),
    (2, [], False),
])
def test_resume_requires_consistent_policy_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_step: object, progress: object, valid: bool,
) -> None:
    """Check progress before invoking any distributed load or policy publication."""
    (tmp_path / "rank_0").mkdir()
    (tmp_path / "checkpoint_complete.json").write_text(json.dumps({
        "world_size": 1, "step": manifest_step,
    }), encoding="utf-8")
    (tmp_path / "extra_state.json").write_text(json.dumps(progress), encoding="utf-8")
    monkeypatch.setattr("rl.checkpoint.dist.get_world_size", lambda: 1)
    monkeypatch.setattr("rl.checkpoint.dist.get_rank", lambda: 0)
    manager = RLCheckpointManager(SimpleNamespace(),
                                  {"output_dir": str(tmp_path), "load_path": str(tmp_path)}, {},
                                  lambda _operation, callback: callback())
    if valid:
        manager.validate_resume()
    else:
        with pytest.raises(RuntimeError, match="progress must be an object|global_step must match"):
            manager.validate_resume()
