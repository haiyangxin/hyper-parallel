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
"""Ensure model-facing SWE-bench data contains only public issues and identities."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq

from examples.code_agent.swebench_data import adapt_row, prepare_data


class TestSWEBenchData(unittest.TestCase):
    """Check private registry fields cannot enter generated model prompts."""

    def test_public_projection_and_identity(self) -> None:
        """Private fields stay in the registry while its exact bytes bind the task."""
        instance = {"instance_id": "pytest-dev__pytest-10051", "problem_statement": "Public issue text",
                    "patch": "PRIVATE_GOLD", "test_patch": "PRIVATE_TEST", "hints_text": "PRIVATE_HINT",
                    "FAIL_TO_PASS": ["PRIVATE_F2P"], "PASS_TO_PASS": ["PRIVATE_P2P"]}
        registry = {"schema_version": 1, "instances": {instance["instance_id"]: {"instance": instance}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = json.dumps(registry).encode()
            (root / "registry.json").write_bytes(raw)
            prepare_data(root / "registry.json", root / "tasks.parquet")
            rows = pq.read_table(root / "tasks.parquet").to_pylist()
        self.assertEqual(len(rows), 1)
        self.assertNotIn("PRIVATE_", json.dumps(rows))
        prompt = adapt_row(rows[0], 0)
        self.assertIn("Public issue text", prompt.messages[0].content)
        self.assertEqual(prompt.ground_truth, {"instance_id": instance["instance_id"],
                                            "registry_sha256": hashlib.sha256(raw).hexdigest()})
        self.assertEqual(prompt.metadata["task_type"], "swebench_verified")
        with self.assertRaises(ValueError):
            adapt_row({**rows[0], "task_id": "different"}, 0)
        with self.assertRaises(ValueError):
            adapt_row({**rows[0], "ground_truth": {**prompt.ground_truth, "patch": "PRIVATE_GOLD"}}, 0)
