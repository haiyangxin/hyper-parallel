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
"""Prepare public SWE-bench prompts while keeping evaluation data controller-side."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from rl.dataset.contracts import Message, PromptRecord


def adapt_row(row: Mapping[str, Any], index: int) -> PromptRecord:
    """Preserve the fixed task identity without copying private registry fields."""
    del index
    truth = row.get("ground_truth")
    if (not isinstance(truth, Mapping) or set(truth) != {"instance_id", "registry_sha256"}
            or not isinstance(truth["instance_id"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(truth["registry_sha256"]))):
        raise ValueError("SWE-bench rows require an instance ID and registry SHA256")
    task_id = "swebench-verified:" + truth["instance_id"]
    if row.get("task_id") != task_id:
        raise ValueError("SWE-bench row identity differs from its registry identity")
    messages = tuple(Message(item["role"], item["content"]) for item in row["prompt"])
    prompt_id = task_id
    metadata = {"task_type": "swebench_verified", "task_id": task_id}
    if "replica_index" in row:
        replica_index = row["replica_index"]
        if isinstance(replica_index, bool) or not isinstance(replica_index, int) or replica_index < 0:
            raise ValueError("SWE-bench replica_index must be a non-negative integer")
        if replica_index > 0:
            prompt_id = f"{task_id}:replica:{replica_index}"
            metadata["replica_index"] = replica_index
    return PromptRecord(prompt_id, messages, dict(truth), metadata)


def prepare_data(registry_path: Path, output: Path, *, repeats: int = 1) -> None:
    """Write public issues with optional sampling identities for repeated instances.

    Args:
        registry_path: Controller-owned instance registry.
        output: Destination parquet file.
        repeats: Number of copies per instance. Repeats provide distributed
            functional coverage, not independent evaluation or a benchmark.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("SWE-bench repeats must be a positive integer")
    raw = registry_path.read_bytes()
    registry = json.loads(raw)
    if registry.get("schema_version") != 1 or not registry.get("instances"):
        raise ValueError("Expected a nonempty SWE-bench registry version 1")
    digest = hashlib.sha256(raw).hexdigest()
    rows = []
    for instance_id, entry in registry["instances"].items():
        instance = entry["instance"]
        issue = instance["problem_statement"]
        if instance["instance_id"] != instance_id or not isinstance(issue, str) or not issue.strip():
            raise ValueError("SWE-bench registry contains an invalid public issue identity")
        rows.append({
            "task_id": "swebench-verified:" + instance_id,
            "ground_truth": {"instance_id": instance_id, "registry_sha256": digest},
            "prompt": [{"role": "user", "content": (
                "Repair the pytest repository in /workspace for the issue below. Inspect source files, "
                "implement a fix and run relevant existing tests. Only Python source under src/_pytest/ "
                "and src/pytest/ may change; preserve tests, configuration and src/_pytest/_version.py. "
                "Use /tmp for scratch files. The repository is offline.\n\n" + issue)}],
        })
    if repeats > 1:
        rows = [{**row, "replica_index": replica_index}
                for replica_index in range(repeats) for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)


def main() -> None:
    """Create model-facing parquet from a controller-owned registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1, help="Copies per instance for distributed functional runs")
    args = parser.parse_args()
    prepare_data(args.registry, args.output, repeats=args.repeats)


if __name__ == "__main__":
    main()
