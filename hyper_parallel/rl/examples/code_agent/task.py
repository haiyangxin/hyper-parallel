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
"""Judge frozen toy repositories in fresh containers with controller-owned expectations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any, Mapping

from rl.agentic.core.types import RewardResult
from rl.agentic.envs.docker_workspace import (
    DockerWorkspace, InvalidSubmissionError, WorkspaceOutputLimitError, WorkspaceTimeoutError, read_snapshot,
)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
_HEALTH_COMMAND = "import json; print(json.dumps({'runtime': 'ready'}, sort_keys=True))"


def content_hash(files: Mapping[str, bytes]) -> str:
    """Identify file paths and bytes independently of candidate Git metadata."""
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def load_fixture(fixture_id: str) -> tuple[dict, dict[str, bytes]]:
    """Resolve a bundled fixture, never a dataset-provided host path."""
    registry = json.loads((ROOT / "tasks.json").read_text(encoding="utf-8"))
    if not isinstance(fixture_id, str) or fixture_id not in registry:
        raise ValueError(f"Unknown repository fixture: {fixture_id!r}")
    directory = ROOT / "fixtures" / fixture_id
    files = {path.relative_to(directory).as_posix(): path.read_bytes()
             for path in directory.rglob("*") if path.is_file()}
    return registry[fixture_id], files


def _manifest(fixture_id: str, fixture: Mapping[str, Any], files: Mapping[str, bytes]) -> dict[str, str]:
    tests = json.dumps(fixture["cases"], sort_keys=True, ensure_ascii=False)
    return {"fixture_id": fixture_id, "base_hash": content_hash(files),
            "test_version": hashlib.sha256(tests.encode()).hexdigest()}


def task_manifest(fixture_id: str) -> dict[str, str]:
    """Return public identity hashes without reference code or hidden tests."""
    fixture, files = load_fixture(fixture_id)
    return _manifest(fixture_id, fixture, files)


def validate_submission(base: Mapping[str, bytes], files: Mapping[str, bytes]) -> dict[str, Any]:
    """Allow direct src/*.py edits only and summarize additions, changes and deletions."""
    changes = {"added": [], "modified": [], "deleted": []}
    for name in sorted(set(base) | set(files)):
        if base.get(name) == files.get(name):
            continue
        path = PurePosixPath(name)
        if (len(path.parts) != 2 or path.parts[0] != "src" or path.suffix != ".py"
                or path.as_posix() != name or "\\" in name):
            raise InvalidSubmissionError(f"Protected path changed: {name}")
        operation = "added" if name not in base else "deleted" if name not in files else "modified"
        changes[operation].append(name)
    return {"content_hash": content_hash(files), "changes": changes,
            "files": {name: hashlib.sha256(value).hexdigest() for name, value in sorted(files.items())}}


def _write_files(directory: Path, files: Mapping[str, bytes]) -> None:
    for name, content in files.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


async def _materialize(directory: Path, files: Mapping[str, bytes]) -> None:
    worker = asyncio.create_task(asyncio.to_thread(_write_files, directory, files))
    try:
        await asyncio.shield(worker)
    except asyncio.CancelledError:
        # Finish bounded local writes before TemporaryDirectory removes their destination.
        await worker
        raise


class RepositoryTask:
    """Prepare public toy sources and award one binary reward for their frozen artifact."""

    def __init__(self, prompt: Any, config: Mapping[str, Any]) -> None:
        """Validate the trusted task identity, pinned image and bounded grading settings."""
        truth = prompt.ground_truth
        if not isinstance(truth, Mapping) or set(truth) != {"fixture_id", "base_hash", "test_version"}:
            raise ValueError("Repository ground_truth must contain only the fixed fixture manifest")
        self.fixture_id = truth["fixture_id"]
        self.fixture, self.base = load_fixture(self.fixture_id)
        if truth != _manifest(self.fixture_id, self.fixture, self.base):
            raise ValueError("Repository fixture or hidden-test identity changed")
        if prompt.metadata.get("task_type") != "repository_python_cli":
            raise ValueError("Repository task requires repository_python_cli metadata")
        allowed = {"image", "memory", "cpus", "pids_limit", "run_timeout", "output_limit_bytes", "python_executable",
                   "run_id"}
        if not isinstance(config, Mapping) or set(config) - allowed:
            raise ValueError("Unsupported repository task settings")
        self.image = config.get("image")
        if not isinstance(self.image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.image):
            raise ValueError("Repository image must be a fixed sha256 image ID")
        self.python = config.get("python_executable", "/usr/local/python3.12.13/bin/python")
        if not isinstance(self.python, str) or not self.python.startswith("/"):
            raise ValueError("Repository python_executable must be an absolute container path")
        self.run_timeout = float(config.get("run_timeout", 5.0))
        self.output_limit = int(config.get("output_limit_bytes", 65536))
        if not 0 < self.run_timeout <= 60 or not 0 < self.output_limit <= 1048576:
            raise ValueError("Repository judge timeout/output budget is out of range")
        self.resources = {key: config[key] for key in ("memory", "cpus", "pids_limit", "run_id") if key in config}
        self.manifest = dict(truth)

    async def prepare(self, workspace: DockerWorkspace) -> None:
        """Copy immutable public baseline bytes, without the private task registry."""
        with tempfile.TemporaryDirectory(prefix="hyper-repository-prepare-") as temp:
            directory = Path(temp)
            await _materialize(directory, self.base)
            await workspace.copy_in(directory)

    async def _case(self, directory: Path, inputs: Any, expected: Any) -> tuple[str, int | None]:
        # A fresh container per case prevents changed tests, runtimes and child processes leaking to later cases.
        grader = DockerWorkspace(self.image, network="none", name_prefix="hyper-repository-grade", **self.resources)
        try:
            await grader.start()
            health = await grader.exec([self.python, "-I", "-c", _HEALTH_COMMAND], timeout=10,
                                       output_limit_bytes=4096)
            if health.returncode or health.stdout.strip() != '{"runtime": "ready"}':
                raise RuntimeError("Repository grader interpreter preflight failed")
            await grader.copy_in(directory)
            try:
                result = await grader.exec([self.python, "/workspace/entrypoint.py"], stdin=json.dumps(inputs),
                                           timeout=self.run_timeout, output_limit_bytes=self.output_limit,
                                           env={"PYTHONDONTWRITEBYTECODE": "1"})
            except (WorkspaceTimeoutError, WorkspaceOutputLimitError) as error:
                status = "timeout" if isinstance(error, WorkspaceTimeoutError) else "output_limit"
                return status, error.result.returncode
            if result.returncode:
                return "runtime_error", result.returncode
            try:
                actual = json.loads(result.stdout)
            except (ValueError, UnicodeError, RecursionError):
                return "wrong_answer", result.returncode
            # Canonical JSON distinguishes boolean values from integer answers.
            passed = json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
            return "passed" if passed else "wrong_answer", result.returncode
        finally:
            original_error = sys.exc_info()[1]
            try:
                await grader.close()
            except Exception:
                if original_error is None:
                    raise
                logger.exception("Grader cleanup failed while preserving original failure")

    async def evaluate(self, archive: Path) -> RewardResult:
        """Score frozen bytes in independent graders; infrastructure errors propagate.

        Invalid submissions and candidate execution failures score zero. Every declared
        case runs, including after a wrong answer, timeout or output limit. No candidate
        success flag or modified test script is used as an oracle.
        """
        metadata = {"task_id": self.fixture_id, "runtime_version": self.image,
                    "base_hash": self.manifest["base_hash"], "test_version": self.manifest["test_version"],
                    "judge_version": "repository-json-v2"}
        try:
            files = await asyncio.to_thread(read_snapshot, archive)
            artifact = validate_submission(self.base, files)
        except InvalidSubmissionError as error:
            return RewardResult(0.0, {"success": 0.0}, {**metadata, "status": "invalid_submission",
                                                      "reason": str(error)})
        artifact_path = archive.with_suffix(".manifest.json")
        await asyncio.to_thread(artifact_path.write_text, json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        metadata.update({"artifact": str(artifact_path), "artifact_hash": artifact["content_hash"]})
        cases = []
        with tempfile.TemporaryDirectory(prefix="hyper-repository-grade-") as temp:
            directory = Path(temp)
            await _materialize(directory, files)
            for index, (inputs, expected) in enumerate(self.fixture["cases"]):
                status, exit_code = await self._case(directory, inputs, expected)
                cases.append({"index": index, "status": status, "exit_code": exit_code})
        passed = sum(case["status"] == "passed" for case in cases)
        success = float(passed == len(cases))
        status = next((case["status"] for case in cases if case["status"] != "passed"), "passed")
        return RewardResult(success, {"success": success, "passed": float(passed), "total": float(len(cases))},
                            {**metadata, "status": status, "evaluated": len(cases), "cases": cases})


def build_task(prompt: Any, config: Mapping[str, Any]) -> RepositoryTask:
    """Build a toy repository evaluator from trusted controller-side settings."""
    return RepositoryTask(prompt, config)
