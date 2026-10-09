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
"""Repeated repository fixtures retain private identities and distinct sampling contexts."""

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq
from rl.agentic.core.program_runner import harness_generation_settings
from rl.dataset.data_source import _DistributedPromptSampler

from examples.code_agent.prepare_data import adapt_row as adapt_toy_row, prepare_data as prepare_toy_data
from examples.code_agent.swebench_data import adapt_row as adapt_swe_row, prepare_data as prepare_swe_data
from examples.code_agent.task import RepositoryTask


class TestCodeAgentData(unittest.TestCase):
    """Check four-rank functional data without changing fixture or grading authority."""

    @staticmethod
    def _registry(root: Path) -> Path:
        """Write public issues and private marker fields to a controller-only registry."""
        registry = {"schema_version": 1, "instances": {}}
        for identity in ("pytest-dev__pytest-10051", "pytest-dev__pytest-10081"):
            registry["instances"][identity] = {"instance": {
                "instance_id": identity, "problem_statement": "Repair the public issue " + identity,
                "patch": "PRIVATE_GOLD", "test_patch": "PRIVATE_TEST", "hints_text": "PRIVATE_HINT",
                "FAIL_TO_PASS": ["PRIVATE_F2P"], "PASS_TO_PASS": ["PRIVATE_P2P"],
            }}
        path = root / "registry.json"
        path.write_text(json.dumps(registry), encoding="utf-8")
        return path

    @classmethod
    def _rows(cls, root: Path, kind: str, repeats: int = 1) -> tuple[list, Callable]:
        """Call the real data writer and return its model-facing rows and adapter."""
        output = root / f"{kind}-{repeats}.parquet"
        if kind == "toy":
            prepare_toy_data(output, repeats=repeats)
            adapter = adapt_toy_row
        else:
            prepare_swe_data(cls._registry(root), output, repeats=repeats)
            adapter = adapt_swe_row
        return pq.read_table(output).to_pylist(), adapter

    def test_default_toy_rows_and_grader_identity_stay_unchanged(self) -> None:
        """The old two-row projection still constructs the original trusted grader tasks."""
        with tempfile.TemporaryDirectory() as directory:
            rows, adapter = self._rows(Path(directory), "toy")
        self.assertEqual([row["task_id"] for row in rows], [
            "repository-v1:merge_intervals", "repository-v1:word_counts",
        ])
        for index, row in enumerate(rows):
            self.assertEqual(set(row), {"task_id", "ground_truth", "prompt"})
            record = adapter(row, index)
            self.assertEqual(record.prompt_id, row["task_id"])
            self.assertEqual(record.metadata, {"task_type": "repository_python_cli", "task_id": row["task_id"]})
            task = RepositoryTask(record, {"image": "sha256:" + "a" * 64})
            self.assertEqual(task.fixture_id, row["ground_truth"]["fixture_id"])
            self.assertEqual(set(record.ground_truth), {"fixture_id", "base_hash", "test_version"})

    def test_default_swe_rows_exclude_private_registry_fields(self) -> None:
        """The registry hash binds public rows without exposing patches or hidden tests."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, adapter = self._rows(root, "swe")
            digest = hashlib.sha256((root / "registry.json").read_bytes()).hexdigest()
        self.assertEqual(len(rows), 2)
        self.assertNotIn("PRIVATE_", json.dumps(rows))
        for index, row in enumerate(rows):
            self.assertEqual(set(row), {"task_id", "ground_truth", "prompt"})
            record = adapter(row, index)
            self.assertEqual(record.prompt_id, row["task_id"])
            self.assertEqual(record.ground_truth, {
                "instance_id": row["ground_truth"]["instance_id"], "registry_sha256": digest,
            })

    def test_repeats_supply_distinct_four_rank_episodes_and_seeds(self) -> None:
        """Every real sampler rank receives a unique context with the original fixture authority."""
        generation = {"max_new_tokens": 1024, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": 20261008}
        with tempfile.TemporaryDirectory() as directory:
            for kind in ("toy", "swe"):
                with self.subTest(kind=kind):
                    default, _ = self._rows(Path(directory), kind)
                    rows, adapter = self._rows(Path(directory), kind, repeats=2)
                    prompts = []
                    for rank in range(4):
                        indices = list(_DistributedPromptSampler(
                            len(rows), rank=rank, world_size=4, seed=1234, shuffle=False,
                        ))
                        self.assertEqual(len(indices), 1)
                        prompts.append(adapter(rows[indices[0]], indices[0]))
                    self.assertEqual(len({prompt.prompt_id for prompt in prompts}), 4)
                    seeds = [harness_generation_settings(generation, prompt.prompt_id, 0, 0)["seed"]
                             for prompt in prompts]
                    self.assertEqual(len(set(seeds)), 4)
                    self.assertEqual([prompt.prompt_id for prompt in prompts[:2]], [row["task_id"] for row in default])
                    for index, prompt in enumerate(prompts):
                        self.assertEqual(prompt.ground_truth, default[index % 2]["ground_truth"])
                        self.assertEqual(prompt.metadata["task_id"], default[index % 2]["task_id"])
                        self.assertEqual(prompt.messages[0].content, default[index % 2]["prompt"][0]["content"])
                    self.assertNotIn("PRIVATE_", json.dumps(rows))

    def test_invalid_repeats_fail_before_writing_data(self) -> None:
        """Boolean, non-integer and non-positive repeat counts cannot create partial datasets."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self._registry(root)
            output = root / "invalid.parquet"
            for value in (0, -1, False, True, None, "2", 1.5):
                for kind in ("toy", "swe"):
                    with self.subTest(kind=kind, repeats=value), self.assertRaisesRegex(ValueError, "positive integer"):
                        if kind == "toy":
                            prepare_toy_data(output, repeats=value)
                        else:
                            prepare_swe_data(registry, output, repeats=value)
                    self.assertFalse(output.exists())

    def test_invalid_replica_indices_do_not_relabel_tasks(self) -> None:
        """Sampling namespaces require an explicit non-negative integer index."""
        with tempfile.TemporaryDirectory() as directory:
            for kind in ("toy", "swe"):
                rows, adapter = self._rows(Path(directory), kind)
                for value in (-1, False, True, None, "1", 1.5):
                    with self.subTest(kind=kind, replica=value), self.assertRaisesRegex(
                        ValueError, "non-negative integer",
                    ):
                        adapter({**rows[0], "replica_index": value}, 0)

    def test_repeated_sampling_identity_cannot_bypass_private_identity_validation(self) -> None:
        """A replica suffix never authorizes a forged fixture, task ID or private field."""
        with tempfile.TemporaryDirectory() as directory:
            for kind in ("toy", "swe"):
                rows, adapter = self._rows(Path(directory), kind, repeats=2)
                for mutation in ("task", "hash", "private"):
                    row = deepcopy(rows[-1])
                    if mutation == "task":
                        row["task_id"] += ":forged"
                    elif mutation == "hash":
                        key = "base_hash" if kind == "toy" else "registry_sha256"
                        row["ground_truth"][key] = "invalid"
                    else:
                        row["ground_truth"]["patch"] = "PRIVATE_GOLD"
                    with self.subTest(kind=kind, mutation=mutation), self.assertRaises(ValueError):
                        adapter(row, 0)
