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
"""Grade fixed SWE-bench instances with independent official environments."""

import asyncio
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import json
import logging
import math
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping

from rl.agentic.core.types import RewardResult
from rl.agentic.envs.docker_workspace import (
    DockerWorkspace, InvalidSubmissionError, WorkspaceOutputLimitError, WorkspaceTimeoutError,
)

from examples.code_agent.swebench_artifacts import build_submission

_LOGGER = logging.getLogger(__name__)
_EVALUATOR = {"version": "4.1.0", "revision": "726c5461e2ef52d83cf1ea2107870a8bb3328d57"}
_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"

_SYNTAX_CHECK = r"""
import hashlib
import json
from pathlib import Path
import sys

if sys.version_info[:2] != (3, 9):
    raise RuntimeError("SWE-bench syntax check requires the task Python 3.9")
result = {"python": sys.version, "executable": sys.executable, "files": {}, "errors": []}
for name in json.loads(sys.stdin.read()):
    data = (Path("/testbed") / name).read_bytes()
    result["files"][name] = hashlib.sha256(data).hexdigest()
    try:
        compile(data, name, "exec", dont_inherit=True)
    except SyntaxError as error:
        result["errors"].append({"path": name, "type": type(error).__name__, "message": error.msg,
                                 "lineno": error.lineno, "offset": error.offset})
print(json.dumps(result))
"""


async def _check_syntax(workspace: DockerWorkspace, paths: list[str]) -> dict:
    result = await workspace.exec([_PYTHON, "-I", "-c", _SYNTAX_CHECK], cwd="/",
                                  stdin=json.dumps(paths), timeout=60)
    if result.returncode:
        raise RuntimeError(f"SWE-bench syntax compiler failed: {result.stderr[:1000]}")
    evidence = json.loads(result.stdout)
    if (set(evidence["files"]) != set(paths) or not evidence["python"].startswith("3.9.")
            or evidence["executable"] != _PYTHON):
        raise RuntimeError("SWE-bench syntax compiler returned mismatched evidence")
    return evidence


def evaluator_source_hashes() -> dict[str, str]:
    """Identify every actual official harness source file, including parsers and grading."""
    spec = find_spec("swebench")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("The pinned SWE-bench evaluator is not installed")
    root = Path(next(iter(spec.submodule_search_locations)))
    files = sorted((root / "harness").rglob("*.py"))
    if not files:
        raise RuntimeError("Official SWE-bench harness source is missing")
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _official() -> tuple:
    # SWE-bench is optional for ordinary RL recipes; require it only for this task.
    from swebench.harness.grading import get_eval_report, get_logs_eval  # pylint: disable=import-outside-toplevel
    from swebench.harness.test_spec.test_spec import make_test_spec  # pylint: disable=import-outside-toplevel

    if version("swebench") != _EVALUATOR["version"]:
        raise RuntimeError("SWE-bench evaluator package version differs from the pinned registry")
    return make_test_spec, get_logs_eval, get_eval_report


async def _checked(workspace: DockerWorkspace, argv: list[str], *, cwd: str = "/") -> str:
    result = await workspace.exec(argv, cwd=cwd, timeout=60)
    if result.returncode:
        raise RuntimeError(f"Trusted repository preparation failed: {result.stderr[:1000]}")
    return result.stdout.strip()


async def _check_base(workspace: DockerWorkspace, base_commit: str) -> None:
    git = ["git", "-c", "safe.directory=/testbed"]
    await _checked(workspace, [*git, "cat-file", "-e", base_commit + "^{commit}"], cwd="/testbed")
    await _checked(workspace, [*git, "merge-base", "--is-ancestor", base_commit, "HEAD"], cwd="/testbed")
    await _checked(workspace, [*git, "diff", "--quiet", base_commit, "--", "src/_pytest", "src/pytest"],
                   cwd="/testbed")


async def prepare_candidate(workspace: DockerWorkspace, base_commit: str) -> None:
    """Expose only this image's public base while keeping editable paths aligned."""
    if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        raise ValueError("SWE-bench base commit must be a full Git object ID")
    await _check_base(workspace, base_commit)
    await _checked(workspace, ["bash", "-ec", "cp -a /testbed/. /workspace/; "
                             "rm -rf /workspace/.git; "
                             r"find /workspace -type d \( -name __pycache__ -o -name .pytest_cache \) "
                             "-prune -exec rm -rf {} +; rm -rf /testbed; ln -s /workspace /testbed"])


class SWEBenchTask:
    """Keep private task data controller-side and replay only frozen source patches."""

    def __init__(self, prompt: Any, config: Mapping[str, Any]) -> None:
        """Validate the trusted registry, baseline archive, evaluator and image identity."""
        allowed = {"registry_path", "run_timeout", "output_limit_bytes", "memory", "cpus", "pids_limit",
                   "run_id", "image"}
        if set(config) - allowed:
            raise ValueError("Unsupported SWE-bench task settings")
        truth = prompt.ground_truth
        if (not isinstance(truth, Mapping) or set(truth) != {"instance_id", "registry_sha256"}
                or prompt.metadata.get("task_type") != "swebench_verified"):
            raise ValueError("SWE-bench prompt must carry only the trusted registry identity")
        registry_path = Path(config["registry_path"]).resolve()
        raw = registry_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != truth["registry_sha256"]:
            raise ValueError("SWE-bench registry identity changed")
        registry = json.loads(raw)
        if registry.get("schema_version") != 1 or registry.get("evaluator") != _EVALUATOR:
            raise ValueError("Unsupported SWE-bench registry/evaluator version")
        self.instance_id = truth["instance_id"]
        entry = registry["instances"][self.instance_id]
        self.instance = entry["instance"]
        if self.instance["instance_id"] != self.instance_id:
            raise ValueError("SWE-bench instance identity differs from its registry key")
        self.workspace_image = entry["image"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.workspace_image):
            raise ValueError("SWE-bench image must be a fixed sha256 image ID")
        self.baseline = (registry_path.parent / entry["baseline_archive"]).resolve()
        if not self.baseline.is_relative_to(registry_path.parent):
            raise ValueError("Baseline archive must stay inside its registry directory")
        if hashlib.sha256(self.baseline.read_bytes()).hexdigest() != entry["baseline_sha256"]:
            raise ValueError("SWE-bench baseline archive identity changed")
        if registry.get("evaluator_source_hashes") != evaluator_source_hashes():
            raise RuntimeError("Official SWE-bench evaluator source differs from the pinned registry")
        self.make_spec, self.parse_log, self.grade_report = _official()
        self.spec = self.make_spec(self.instance, arch="arm64", namespace=None)
        self.eval_script = self.spec.eval_script
        if hashlib.sha256(self.eval_script.encode()).hexdigest() != entry["eval_script_sha256"]:
            raise RuntimeError("Official SWE-bench evaluation script differs from the pinned registry")
        self.timeout = float(config.get("run_timeout", 300))
        self.output_limit = int(config.get("output_limit_bytes", 8 * 1024 * 1024))
        if not math.isfinite(self.timeout) or self.timeout <= 0 or self.output_limit <= 0:
            raise ValueError("SWE-bench judge budgets must be positive and finite")
        self.resources = {key: config[key] for key in ("memory", "cpus", "pids_limit", "run_id") if key in config}

    async def prepare(self, workspace: DockerWorkspace) -> None:
        """Prepare public tools before any candidate model execution."""
        if workspace.image != self.workspace_image:
            raise RuntimeError("Candidate image differs from the pinned SWE-bench task image")
        await prepare_candidate(workspace, self.instance["base_commit"])

    async def _run_grader(self, patch: str, log_path: Path, changes: Mapping[str, list[str]]) -> dict | None:
        grader = DockerWorkspace(self.workspace_image, network="none", name_prefix="hyper-swe-grade", **self.resources)
        try:
            await grader.start()
            await _check_base(grader, self.instance["base_commit"])
            await _checked(grader, [_PYTHON, "-c", "import pytest; print(pytest.__version__)"])
            with tempfile.TemporaryDirectory(prefix="hyper-swe-grade-") as directory:
                root = Path(directory)
                (root / "submission.patch").write_text(patch, encoding="utf-8")
                (root / "eval.sh").write_text(self.eval_script, encoding="utf-8")
                await grader.copy_in(root)
            baseline_syntax = await _check_syntax(grader, sorted(changes["modified"] + changes["deleted"]))
            if baseline_syntax["errors"]:
                raise RuntimeError("Trusted SWE-bench baseline source has invalid syntax")
            if patch:
                git = ["git", "-c", "safe.directory=/testbed", "apply"]
                await _checked(grader, [*git, "--check", "/workspace/submission.patch"], cwd="/testbed")
                await _checked(grader, [*git, "/workspace/submission.patch"], cwd="/testbed")
            candidate_syntax = await _check_syntax(grader, sorted(changes["added"] + changes["modified"]))
            if candidate_syntax["errors"]:
                log_path.write_text("Official tests not run: independently verified candidate syntax error.\n",
                                    encoding="utf-8")
                return {"baseline": baseline_syntax, "candidate": candidate_syntax}
            try:
                result = await grader.exec(["bash", "-c", "exec bash /workspace/eval.sh 2>&1"], cwd="/testbed",
                                           timeout=self.timeout, output_limit_bytes=self.output_limit)
            except (WorkspaceTimeoutError, WorkspaceOutputLimitError) as error:
                log_path.write_text(error.result.stdout, encoding="utf-8")
                raise
            log_path.write_text(result.stdout, encoding="utf-8")
            return None
        finally:
            original_error = sys.exc_info()[1]
            try:
                await grader.close()
            except Exception:
                if original_error is None:
                    raise
                _LOGGER.exception("SWE-bench grader cleanup failed while preserving the original failure")

    async def evaluate(self, archive: Path) -> RewardResult:
        """Use official resolved status or proven syntax-error zero; uncertain failures reject."""
        metadata = {"instance_id": self.instance_id, "image": self.workspace_image,
                    "base_commit": self.instance["base_commit"], "judge_version": _EVALUATOR["revision"]}
        try:
            worker = asyncio.create_task(asyncio.to_thread(build_submission, archive, self.baseline))
            try:
                artifact = await asyncio.shield(worker)
            except asyncio.CancelledError:
                await asyncio.gather(worker, return_exceptions=True)
                raise
        except InvalidSubmissionError as error:
            return RewardResult(0., {"success": 0.}, {**metadata, "status": "invalid_submission", "reason": str(error)})
        manifest_path = archive.with_suffix(".manifest.json")
        patch = artifact.pop("patch")
        manifest_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        archive.with_suffix(".patch").write_text(patch, encoding="utf-8")
        log_path = archive.with_suffix(".eval.log")
        syntax = await self._run_grader(patch, log_path, artifact["changes"])
        if syntax is not None:
            syntax.update(artifact_hash=artifact["content_hash"], patch_sha256=artifact["patch_sha256"])
            syntax_path = archive.with_suffix(".syntax.json")
            syntax_path.write_text(json.dumps(syntax, indent=2) + "\n", encoding="utf-8")
            return RewardResult(0., {"success": 0.}, {
                **metadata, "status": "syntax_error", "failure_origin": "model", "trainable": True,
                "artifact": str(manifest_path), "artifact_hash": artifact["content_hash"],
                "patch_sha256": artifact["patch_sha256"], "syntax_path": str(syntax_path), "evaluated": 0})
        prediction = {"instance_id": self.instance_id, "model_patch": patch, "model_name_or_path": "hyper-rl"}
        report = self.grade_report(test_spec=self.spec, prediction=prediction,
                                   test_log_path=str(log_path), include_tests_status=True)
        report_path = archive.with_suffix(".report.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        statuses, found = self.parse_log(self.spec, str(log_path))
        required = set(self.spec.FAIL_TO_PASS) | set(self.spec.PASS_TO_PASS)
        if (not found or not required or not required.issubset(statuses)
                or any(statuses[name] not in {"PASSED", "XFAIL", "FAILED", "ERROR"} for name in required)):
            raise RuntimeError("Official evaluation did not cover every required F2P/P2P test with a valid status")
        info = report[self.instance_id]
        if not info.get("patch_successfully_applied") or not isinstance(info.get("resolved"), bool):
            raise RuntimeError("Official evaluation report is incomplete")
        success = float(info["resolved"])
        metadata.update({"status": "resolved" if success else "unresolved", "artifact": str(manifest_path),
                         "artifact_hash": artifact["content_hash"], "patch_sha256": artifact["patch_sha256"],
                         "report_path": str(report_path), "evaluated": len(required)})
        return RewardResult(success, {"success": success}, metadata)


def build_task(prompt: Any, config: Mapping[str, Any]) -> SWEBenchTask:
    """Construct one instance evaluator from the controller-owned fixed registry."""
    return SWEBenchTask(prompt, config)
