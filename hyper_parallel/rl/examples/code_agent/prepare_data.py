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
"""Prepare two public repository tasks with opaque grading identity hashes."""

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
    return PromptRecord(task_id, tuple(Message(item["role"], item["content"]) for item in row["prompt"]),
                        dict(manifest), {"task_type": "repository_python_cli", "task_id": task_id})


def prepare_data(output: Path) -> None:
    """Write functional toy tasks; this overlapping two-row set is not a benchmark."""
    rows = []
    for fixture_id in ("merge_intervals", "word_counts"):
        fixture, _ = load_fixture(fixture_id)
        rows.append({"task_id": f"repository-v1:{fixture_id}", "ground_truth": task_manifest(fixture_id),
                     "prompt": [{"role": "user", "content": (
                         "Fix the Python repository in /workspace. Read its files, modify source code, and run "
                         "python public_test.py before finishing. Preserve protected files. "
                         + fixture["description"])}]})
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)


def main() -> None:
    """Write the prepared toy task parquet file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare_data(parser.parse_args().output)


if __name__ == "__main__":
    main()
