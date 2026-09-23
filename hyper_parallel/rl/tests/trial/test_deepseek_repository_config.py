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
"""DeepSeek repository configuration rejects unsafe candidate contracts."""

from copy import deepcopy

import pytest

from rl.config import _validate_deepseek_agentic, _validate_external_agentic_rollout


_IMAGE = "sha256:" + "a" * 64
_CONFIG = {"version": "0.1.1rc1", "provider": "deepseek-official", "model": "policy",
           "session_root": "/tmp/deepseek-repository", "gateway_port": 8300,
           "timeout_seconds": 30, "request_timeout": 10,
           "task_factory": "examples.code_agent.task:build_task", "workspace": {"image": _IMAGE}}


def test_deepseek_repository_config_and_text_reward_remain_exclusive() -> None:
    _validate_deepseek_agentic({"deepseek": _CONFIG})
    text = deepcopy(_CONFIG)
    text.pop("task_factory")
    text.pop("workspace")
    text["reward_callable"] = "examples.gsm8k.agent:score_deepseek_gsm8k_answer"
    _validate_deepseek_agentic({"deepseek": text})
    for bad in (None, "", "no-colon", "a:b:c"):
        invalid = {**_CONFIG, "task_factory": bad}
        with pytest.raises(ValueError):
            _validate_deepseek_agentic({"deepseek": invalid})
    with pytest.raises(ValueError, match="exactly one"):
        _validate_deepseek_agentic({"deepseek": {**_CONFIG, "reward_callable": text["reward_callable"]}})


@pytest.mark.parametrize("override", [
    {"workspace": {"image": "mutable:tag"}},
    {"workspace": {"image": _IMAGE, "network": "host"}},
    {"workspace": {"image": _IMAGE, "max_concurrent": 0}},
    {"task_config": {"image": "sha256:" + "b" * 64}},
    {"workspace_template": "/host/source"},
    {"mcp_servers": [{"name": "host-tool"}]},
    {"runtime_bin": "/host/binary"},
])
def test_deepseek_repository_rejects_unsafe_workspace(override: dict) -> None:
    with pytest.raises(ValueError):
        _validate_deepseek_agentic({"deepseek": {**_CONFIG, **override}})


def test_deepseek_repository_requires_room_for_model_output() -> None:
    rollout = {"max_new_tokens": 1024,
               "vllm": {"max_model_len": 1536, "port": 8301, "logprobs_mode": "raw_logprobs",
                        "enable_auto_tool_choice": True, "tool_call_parser": "hermes"}}
    with pytest.raises(ValueError, match="context room"):
        _validate_external_agentic_rollout("deepseek", "vllm", rollout, {"deepseek": _CONFIG})
    rollout["vllm"]["max_model_len"] = 16384
    _validate_external_agentic_rollout("deepseek", "vllm", rollout, {"deepseek": _CONFIG})
