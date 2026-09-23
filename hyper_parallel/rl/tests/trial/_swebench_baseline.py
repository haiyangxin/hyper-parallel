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
"""Run fixed SWE-bench public-baseline and reference-patch official controls."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace

from rl.agentic.envs.docker_workspace import DockerWorkspace, _fixture_archive, read_snapshot

from examples.code_agent.swebench_data import prepare_data
from examples.code_agent.swebench_task import SWEBenchTask, _official, evaluator_source_hashes, prepare_candidate


def _gold_archive(baseline: Path, patch: str, destination: Path) -> None:
    """Apply only a trusted reference source patch to controller-owned frozen bytes."""
    files = read_snapshot(baseline)
    with tarfile.open(baseline, "r:") as archive:
        modes = {str(Path(member.name)): member.mode & 0o777 for member in archive if member.isfile()}
    with tempfile.TemporaryDirectory(prefix="hyper-swe-reference-") as directory:
        root = Path(directory)
        for name, content in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            target.chmod(modes[name])
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull})
        for mode in (["--check"], []):
            subprocess.run(["git", "apply", *mode, "-"], input=patch, text=True, cwd=root, env=env,
                           capture_output=True, check=True, timeout=30)
        destination.write_bytes(_fixture_archive(root))


async def _baseline(instance: dict, image: str, destination: Path, run_id: str) -> None:
    workspace = DockerWorkspace(image, network="none", memory="4g", cpus=2, pids_limit=256,
                                run_id=run_id, name_prefix="hyper-swe-baseline")
    try:
        await workspace.start()
        await prepare_candidate(workspace, instance["base_commit"])
        await workspace.stop()
        await workspace.export(destination)
    finally:
        await workspace.close()


def verify_controls(output: Path) -> None:
    """Audit exact F2P/P2P membership, not merely expected binary control rewards."""
    registry = json.loads((output / "registry.json").read_text())
    checked = []
    for identity, entry in registry["instances"].items():
        instance = entry["instance"]
        expected = {key: set(json.loads(instance[key]) if isinstance(instance[key], str) else instance[key])
                    for key in ("FAIL_TO_PASS", "PASS_TO_PASS")}
        for label in ("baseline", "reference"):
            report = json.loads((output / identity / f"{label}.report.json").read_text())[identity]
            for bucket, required in expected.items():
                status = report["tests_status"][bucket]
                passed, failed = set(status["success"]), set(status["failure"])
                should_pass = label == "reference" or bucket == "PASS_TO_PASS"
                if ((passed != required or failed) if should_pass else (failed != required or passed)):
                    raise RuntimeError(f"{identity} {label} {bucket} differs from the required control outcome")
            checked.append({"instance_id": identity, "control": label,
                            "F2P": len(expected["FAIL_TO_PASS"]), "P2P": len(expected["PASS_TO_PASS"])})
    (output / "controls-audit.json").write_text(json.dumps({"status": "passed", "controls": checked}, indent=2) + "\n")


async def run(args: argparse.Namespace) -> None:
    """Prepare identity-pinned controls then run all official selected tests."""
    images = json.loads(args.images.read_text())
    rows = {row["instance_id"]: row for row in json.loads(args.instances.read_text())}
    args.output.mkdir(parents=True, exist_ok=True)
    make_spec, _, _ = _official()
    registry = {"schema_version": 1,
                "dataset": {"id": "princeton-nlp/SWE-bench_Verified",
                            "revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a", "split": "test"},
                "evaluator": {"version": "4.1.0", "revision": "726c5461e2ef52d83cf1ea2107870a8bb3328d57"},
                "evaluator_source_hashes": evaluator_source_hashes(), "instances": {}}
    for identity, image in images.items():
        instance = rows[identity]
        directory = args.output / identity
        directory.mkdir(exist_ok=True)
        baseline = directory / "baseline.tar"
        await _baseline(instance, image, baseline, args.run_id)
        spec = make_spec(instance, arch="arm64", namespace=None)
        registry["instances"][identity] = {"instance": instance, "image": image,
            "baseline_archive": str(baseline.relative_to(args.output)),
            "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
            "eval_script_sha256": hashlib.sha256(spec.eval_script.encode()).hexdigest()}
    registry_path = args.output / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    prepare_data(registry_path, args.output / "tasks.parquet")
    registry_hash = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    results = []
    for identity, entry in registry["instances"].items():
        prompt = SimpleNamespace(ground_truth={"instance_id": identity, "registry_sha256": registry_hash},
                                 metadata={"task_type": "swebench_verified"})
        task = SWEBenchTask(prompt, {"registry_path": str(registry_path), "run_timeout": args.timeout,
                                    "memory": "4g", "cpus": 2, "pids_limit": 256, "run_id": args.run_id})
        baseline = args.output / entry["baseline_archive"]
        gold = baseline.with_name("reference.tar")
        await asyncio.to_thread(_gold_archive, baseline, entry["instance"]["patch"], gold)
        for label, archive, expected in (("original", baseline, 0.), ("reference", gold, 1.)):
            reward = await task.evaluate(archive)
            record = {"instance_id": identity, "control": label, "reward": reward.value,
                      "expected": expected, "metadata": reward.metadata}
            results.append(record)
            (args.output / "controls.json").write_text(json.dumps(results, indent=2) + "\n")
            if reward.value != expected:
                raise RuntimeError(f"{identity} {label} control produced {reward.value}, expected {expected}")
    verify_controls(args.output)
    (args.output / "completed.json").write_text(json.dumps({"status": "passed", "registry_sha256": registry_hash,
        "controls": results, "boundary": "Reference controls only; no model or training evidence"}, indent=2) + "\n")


def main() -> None:
    """Run explicitly; this CPU Docker experiment is not an automatically collected test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_controls(args.output)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
