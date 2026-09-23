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
"""Explicit CPU Docker acceptance for frozen repositories; no model or NPU is used."""

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

from rl.agentic.envs.docker_workspace import (
    DockerWorkspace, InvalidSubmissionError, WorkspaceOutputLimitError, WorkspaceTimeoutError,
)

from examples.code_agent.prepare_data import adapt_row, prepare_data
from examples.code_agent.task import RepositoryTask, load_fixture, task_manifest


PYTHON = "/usr/local/python3.12.13/bin/python"
WRITE_CHANGES = """import json, pathlib, sys
for name, content in json.load(sys.stdin).items():
    path = pathlib.Path('/workspace') / name
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
"""


def require(condition: bool, message: str) -> None:
    """Fail acceptance without relying on optimized-away assertions."""
    if not condition:
        raise RuntimeError(message)


async def inspect_container(name: str) -> dict:
    """Read actual Docker isolation and lifecycle state."""
    result = await asyncio.to_thread(subprocess.run, ["docker", "inspect", name],
                                     capture_output=True, text=True, check=True, timeout=30)
    return json.loads(result.stdout)[0]


def make_task(fixture_id: str, image: str, run_id: str) -> RepositoryTask:
    """Use the production row adapter and fixed grader identity."""
    prompt = adapt_row({"task_id": f"repository-v1:{fixture_id}", "ground_truth": task_manifest(fixture_id),
                        "prompt": [{"role": "user", "content": "Repair this repository."}]}, 0)
    return RepositoryTask(prompt, {"image": image, "run_id": run_id})


async def baseline(args: argparse.Namespace, fixture_id: str, repaired: bool) -> dict:
    """Run a real public test, freeze source and grade independently."""
    task = make_task(fixture_id, args.image, args.run_id)
    workspace = DockerWorkspace(args.image, run_id=args.run_id)
    try:
        await workspace.start()
        await task.prepare(workspace)
        info = await inspect_container(workspace.name)
        host = info["HostConfig"]
        require(not info["Mounts"] and not host["Privileged"] and not host.get("Devices"), "Unsafe mounts/devices")
        require(host["NetworkMode"] == "none" and host["CapDrop"] == ["ALL"], "Isolation options differ")
        require("no-new-privileges" in host["SecurityOpt"], "Missing no-new-privileges")
        require(host["Memory"] == 1024**3 and host["PidsLimit"] == 64 and host["NanoCpus"] == 10**9,
                "Resource limits differ")
        started = time.monotonic()
        shell = await workspace.exec(["/bin/bash", "-lc", "codex --version && python --version"], timeout=10)
        require(shell.returncode == 0 and "codex-cli 0.152.1" in shell.stdout, "Candidate CLI/profile is invalid")
        shell_seconds = time.monotonic() - started
        helper = await workspace.exec([PYTHON, "-c", """import os, pathlib, subprocess
home = pathlib.Path(os.environ['HOME'])
codex_home = pathlib.Path(os.environ['CODEX_HOME'])
(home / 'writable').write_text('ok')
(codex_home / 'writable').write_text('ok')
assert pathlib.Path('/workspace/src/solution.py').stat().st_uid == 0
link = codex_home / 'apply_patch'
link.symlink_to('/opt/codex/bin/codex')
patch = '*** Begin Patch\\n*** Add File: /opt/hyper-agent-home/patched\\n+helper-ok\\n*** End Patch'
subprocess.run([str(link), patch], check=True, timeout=5)
assert (home / 'patched').read_text() == 'helper-ok\\n'
"""], timeout=10)
        require(helper.returncode == 0, "Candidate HOME, ownership or CLI patch helper is invalid")
        cli = await workspace.exec(["codex", "exec", "--help"], timeout=10)
        require(cli.returncode == 0, "Candidate Codex exec cannot start")
        if repaired:
            fixture, _ = load_fixture(fixture_id)
            changes = dict(fixture["reference"])
            changes["src/acceptance_added.py"] = '"""Exercise added source export."""\n'
            if fixture_id == "word_counts":
                changes["src/legacy.py"] = None
            changed = await workspace.exec([PYTHON, "-c", WRITE_CHANGES], stdin=json.dumps(changes))
            require(changed.returncode == 0, "Candidate source edit failed")
        public = await workspace.exec([PYTHON, "public_test.py"])
        if repaired:
            require(public.returncode == 0, "Reference patch failed public test")
        try:
            await workspace.export(args.output / "must-not-export-live.tar")
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Live workspace export was accepted")
        await workspace.stop()
        archive = await workspace.export(args.output / f"{fixture_id}-{'reference' if repaired else 'broken'}.tar")
        reward = await task.evaluate(archive)
        require(reward.value == float(repaired), f"Unexpected reward: {reward}")
        require(reward.metadata["evaluated"] == 5, "Not every hidden case was evaluated")
        return {"fixture": fixture_id, "reference": repaired, "reward": asdict(reward),
                "public_exit": public.returncode, "shell_seconds": shell_seconds,
                "cli": shell.stdout.strip(), "patch_helper_executed": True,
                "isolation": {key: host[key] for key in (
                    "NetworkMode", "Memory", "PidsLimit", "NanoCpus", "CapDrop", "SecurityOpt")}}
    finally:
        await workspace.close()
        await workspace.close()


async def failure_case(args: argparse.Namespace, kind: str) -> None:
    """Exercise real submission, execution-budget, daemon and cancellation failures."""
    workspace = DockerWorkspace(args.image, run_id=args.run_id)
    task = make_task("merge_intervals", args.image, args.run_id)
    try:
        await workspace.start()
        await task.prepare(workspace)
        if kind == "protected":
            await workspace.exec([PYTHON, "-c", WRITE_CHANGES], stdin=json.dumps({"entrypoint.py": "print(True)\n"}))
            await workspace.stop()
            archive = await workspace.export(args.output / "protected.tar")
            reward = await task.evaluate(archive)
            require(reward.value == 0 and reward.metadata["status"] == "invalid_submission", "Protected edit accepted")
            return
        if kind == "symlink":
            await workspace.exec(["ln", "-s", "/etc/passwd", "/workspace/src/escape.py"])
            await workspace.stop()
            try:
                await workspace.export(args.output / "symlink.tar")
            except InvalidSubmissionError:
                return
            raise RuntimeError("Linked submission accepted")
        if kind == "missing_command":
            try:
                await workspace.exec(["/missing-executable"])
            except RuntimeError:
                return
            raise RuntimeError("Missing executable was not attributed to infrastructure")
        if kind == "cancel":
            pending = asyncio.create_task(workspace.exec([PYTHON, "-c", "import time; time.sleep(60)"]))
            await asyncio.sleep(0.3)
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                pass
            else:
                raise RuntimeError("Cancellation was lost")
        else:
            code = "import time; time.sleep(60)" if kind == "timeout" else "print('x' * 100000)"
            expected = WorkspaceTimeoutError if kind == "timeout" else WorkspaceOutputLimitError
            try:
                await workspace.exec([PYTHON, "-c", code], timeout=0.3 if kind == "timeout" else 10,
                                     output_limit_bytes=4096)
            except expected:
                pass
            else:
                raise RuntimeError(f"Missing {kind} failure")
        require(not (await inspect_container(workspace.name))["State"]["Running"], f"{kind} left processes running")
    finally:
        await workspace.close()


async def run(args: argparse.Namespace) -> None:
    """Write results only after every declared acceptance check and owned-container scan passes."""
    args.output.mkdir(parents=True, exist_ok=True)
    prepare_data(args.output / "tasks.parquet")
    results = []
    for fixture_id in ("merge_intervals", "word_counts"):
        for repaired in (False, True):
            results.append(await baseline(args, fixture_id, repaired))
    failures = ["protected", "symlink", "missing_command", "timeout", "output", "cancel"]
    for kind in failures:
        await failure_case(args, kind)
    remaining = await asyncio.to_thread(
        subprocess.run, ["docker", "ps", "-aq", "--filter", "label=hyper-rl.owner=code-agent",
                         "--filter", f"label=hyper-rl.run-id={args.run_id}"],
        capture_output=True, text=True, check=True, timeout=30)
    require(not remaining.stdout.strip(), f"Owned containers remain: {remaining.stdout}")
    report = {"status": "passed", "image": args.image, "run_id": args.run_id, "baselines": results,
              "failure_checks": failures, "remaining_owned_containers": [], "npu_count": 0}
    (args.output / "completed.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "baselines": len(results), "failure_checks": len(failures)}))


def main() -> None:
    """Parse explicit image/output/run ownership for a real Docker experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
