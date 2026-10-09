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
"""Prepare public toy repository tasks with opaque grading identity hashes."""

import argparse
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from rl.dataset.contracts import Message, PromptRecord
from examples.code_agent.task import load_fixture, task_manifest


def adapt_row(row: Mapping[str, Any], index: int) -> PromptRecord:
    """Keep hidden expectations in the controller registry, outside prompt messages."""
    del index
    manifest = row.get("ground_truth")
    if not isinstance(manifest, Mapping) or manifest != task_manifest(manifest.get("fixture_id")):
        raise ValueError("Repository dataset manifest differs from the trusted fixture")
    task_id = f"repository-v1:{manifest['fixture_id']}"
    if row.get("task_id") != task_id:
        raise ValueError("Repository task identity differs from its fixture")
    prompt_id = task_id
    metadata = {"task_type": "repository_python_cli", "task_id": task_id}
    if "replica_index" in row:
        replica_index = row["replica_index"]
        if isinstance(replica_index, bool) or not isinstance(replica_index, int) or replica_index < 0:
            raise ValueError("Repository replica_index must be a non-negative integer")
        if replica_index > 0:
            prompt_id = f"{task_id}:replica:{replica_index}"
            metadata["replica_index"] = replica_index
    return PromptRecord(prompt_id, tuple(Message(item["role"], item["content"]) for item in row["prompt"]),
                        dict(manifest), metadata)


def prepare_data(output: Path, *, repeats: int = 1) -> None:
    """Write toy tasks with optional distinct sampling identities for repeated fixtures.

    Args:
        output: Destination parquet file.
        repeats: Number of copies per fixture. Repeated tasks provide distributed
            functional coverage, not independent evaluation or a benchmark.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("Repository repeats must be a positive integer")
    rows = []
    for replica_index in range(repeats):
        for fixture_id in ("merge_intervals", "word_counts"):
            fixture, _ = load_fixture(fixture_id)
            row = {"task_id": f"repository-v1:{fixture_id}", "ground_truth": task_manifest(fixture_id),
                   "prompt": [{"role": "user", "content": (
                       "Fix the Python repository in /workspace. Read its files, modify source code, and run "
                       "python public_test.py before finishing. Preserve protected files. "
                       + fixture["description"])}]}
            if repeats > 1:
                row["replica_index"] = replica_index
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)


def main() -> None:
    """Write the prepared toy task parquet file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1, help="Copies per toy fixture for distributed functional runs")
    args = parser.parse_args()
    prepare_data(args.output, repeats=args.repeats)


if __name__ == "__main__":
    main()
