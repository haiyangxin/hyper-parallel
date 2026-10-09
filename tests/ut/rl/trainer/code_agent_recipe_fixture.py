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
"""Shared local model identity and public data for repository recipe CPU tests."""

import json
from pathlib import Path
import tempfile
import unittest

import yaml

import rl
from examples.code_agent.prepare_data import prepare_data as prepare_toy_data
from examples.code_agent.swebench_data import prepare_data as prepare_swe_data


def prepare_code_agent_fixture(test_case: unittest.TestCase) -> Path:
    """Create local model/data fixtures whose lifetime belongs to one test case.

    Args:
        test_case: Owner of the temporary directory cleanup.

    Returns:
        Directory containing the hybrid model identity and repeated public tasks.
    """
    # The test case retains cleanup so these files survive until its assertions finish.
    directory = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    test_case.addCleanup(directory.cleanup)
    root = Path(directory.name)
    (root / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
        "text_config": {"model_type": "qwen3_5_text", "tie_word_embeddings": False,
                        "layer_types": ["linear_attention", "full_attention"]},
    }), encoding="utf-8")
    prepare_toy_data(root / "code_agent.parquet", repeats=2)
    registry = {"schema_version": 1, "instances": {}}
    for identity in ("pytest-dev__pytest-10051", "pytest-dev__pytest-10081"):
        registry["instances"][identity] = {"instance": {
            "instance_id": identity, "problem_statement": "Public CPU projection fixture " + identity,
        }}
    (root / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    prepare_swe_data(root / "registry.json", root / "swebench.parquet", repeats=2)
    return root


def load_code_agent_recipe(root: Path, model: str, task: str) -> dict:
    """Read a production recipe while replacing only its external model/data paths.

    Args:
        root: Directory produced by prepare_code_agent_fixture.
        model: Recipe model prefix, preserving that model's own serving parameters.
        task: Repository task recipe suffix.

    Returns:
        Production configuration bound to the local test fixtures.
    """
    recipes = Path(rl.__file__).resolve().parent.parent / "examples/code_agent/configs"
    config = yaml.safe_load((recipes / f"{model}_{task}.yaml").read_text(encoding="utf-8"))
    config["model"].update(weights_path=str(root), tokenizer_path=str(root))
    data_path = str(root / f"{task}.parquet")
    config["data"].update(train_path=data_path, test_path=data_path)
    if task == "swebench":
        config["agentic"]["codex"]["task_config"]["registry_path"] = str(root / "registry.json")
    return config
