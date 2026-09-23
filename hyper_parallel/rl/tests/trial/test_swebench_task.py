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
"""Official evaluator provenance, coverage and isolated grader orchestration."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from rl.agentic.envs.docker_workspace import CommandResult, InvalidSubmissionError

from examples.code_agent import swebench_task


class TestSWEBenchTask(unittest.IsolatedAsyncioTestCase):
    """Mock external APIs while retaining task identity and failure contracts."""

    def setUp(self) -> None:
        """Create a private registry and fixed evaluator response."""
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        baseline = self.directory / "base.tar"
        baseline.write_bytes(b"baseline")
        self.spec = SimpleNamespace(eval_script="official script", FAIL_TO_PASS=["f"], PASS_TO_PASS=["p"])
        self.report = {"instance": {"resolved": False, "patch_successfully_applied": True,
                                     "tests_status": {"FAIL_TO_PASS": {"failure": ["f"]}}}}
        self.grade = MagicMock(return_value=self.report)
        self.parse = MagicMock(return_value=({"f": "FAILED", "p": "PASSED"}, True))
        official = patch.object(swebench_task, "_official", return_value=(MagicMock(return_value=self.spec),
                                                                        self.parse, self.grade))
        official.start()
        self.addCleanup(official.stop)
        hashes = patch.object(swebench_task, "evaluator_source_hashes", return_value={"harness/grading.py": "fixed"})
        hashes.start()
        self.addCleanup(hashes.stop)
        self.image = "sha256:" + "a" * 64
        registry = {"schema_version": 1, "evaluator": dict(swebench_task._EVALUATOR),
                    "evaluator_source_hashes": {"harness/grading.py": "fixed"},
                    "instances": {"instance": {"instance": {"instance_id": "instance", "base_commit": "b" * 40},
                                               "image": self.image, "baseline_archive": "base.tar",
                                               "baseline_sha256": hashlib.sha256(b"baseline").hexdigest(),
                                               "eval_script_sha256": hashlib.sha256(b"official script").hexdigest()}}}
        self.registry = self.directory / "registry.json"
        self.registry.write_text(json.dumps(registry))
        self.prompt = SimpleNamespace(ground_truth={"instance_id": "instance", "registry_sha256": hashlib.sha256(
            self.registry.read_bytes()).hexdigest()}, metadata={"task_type": "swebench_verified"})
        self.task = swebench_task.SWEBenchTask(self.prompt, {"registry_path": str(self.registry)})
        self.archive = self.directory / "submission.tar"
        self.artifact = {"patch": "", "content_hash": "content", "patch_sha256": "patch",
                         "changes": {"added": [], "modified": [], "deleted": []}}

    async def test_setup_commit_descendant_is_allowed_but_source_drift_rejects(self) -> None:
        """Official setup HEAD may differ while the real base and tracked source remain intact."""
        workspace = SimpleNamespace(exec=AsyncMock(return_value=CommandResult(0, "", "")))
        base = "b" * 40
        await swebench_task._check_base(workspace, base)
        commands = [call.args[0] for call in workspace.exec.call_args_list]
        self.assertIn(["git", "-c", "safe.directory=/testbed", "merge-base", "--is-ancestor", base, "HEAD"], commands)
        self.assertEqual(commands[-1][-5:], ["--quiet", base, "--", "src/_pytest", "src/pytest"])
        workspace.exec.side_effect = [CommandResult(0, "", ""), CommandResult(0, "", ""), CommandResult(1, "", "drift")]
        with self.assertRaisesRegex(RuntimeError, "preparation failed"):
            await swebench_task._check_base(workspace, base)

    async def test_prepare_rejects_wrong_image_before_candidate_copy(self) -> None:
        """No public source preparation runs in a mismatched image."""
        workspace = SimpleNamespace(image="sha256:" + "c" * 64)
        with patch.object(swebench_task, "prepare_candidate", new=AsyncMock()) as prepare:
            with self.assertRaisesRegex(RuntimeError, "Candidate image"):
                await self.task.prepare(workspace)
            prepare.assert_not_awaited()

    async def test_empty_patch_still_grades_and_records_real_report(self) -> None:
        """An empty patch receives official false, never success from an exit code."""
        self.task._run_grader = AsyncMock(return_value=None)
        with patch.object(swebench_task, "build_submission", return_value=dict(self.artifact)):
            reward = await self.task.evaluate(self.archive)
        self.assertEqual(reward.value, 0.)
        self.task._run_grader.assert_awaited_once_with(
            "", self.archive.with_suffix(".eval.log"), self.artifact["changes"])
        self.assertEqual(reward.metadata["evaluated"], 2)
        self.assertEqual(json.loads(self.archive.with_suffix(".report.json").read_text()), self.report)
        self.assertTrue(self.grade.call_args.kwargs["include_tests_status"])

    async def test_resolved_and_missing_or_skipped_coverage(self) -> None:
        """Only a complete official resolved report permits one; incomplete evidence rejects."""
        self.task._run_grader = AsyncMock(return_value=None)
        self.report["instance"]["resolved"] = True
        with patch.object(swebench_task, "build_submission", side_effect=lambda *_: dict(self.artifact)):
            self.parse.return_value = ({"f": "PASSED", "p": "PASSED"}, True)
            self.assertEqual((await self.task.evaluate(self.archive)).value, 1.)
            self.parse.return_value = ({"f": "PASSED", "p": "XFAIL"}, True)
            self.assertEqual((await self.task.evaluate(self.archive)).value, 1.)
            for parsed in (({}, False), ({"f": "PASSED"}, True), ({"f": "PASSED", "p": "SKIPPED"}, True)):
                self.parse.return_value = parsed
                with self.assertRaisesRegex(RuntimeError, "every required"):
                    await self.task.evaluate(self.archive)

    async def test_invalid_submission_zero_but_transport_failure_propagates(self) -> None:
        """Malformed artifacts score zero; Docker and parser errors remain infrastructure."""
        self.task._run_grader = AsyncMock(side_effect=RuntimeError("Docker unavailable"))
        with patch.object(swebench_task, "build_submission", side_effect=InvalidSubmissionError("protected")):
            self.assertEqual((await self.task.evaluate(self.archive)).metadata["status"], "invalid_submission")
        self.task._run_grader.assert_not_awaited()
        with patch.object(swebench_task, "build_submission", return_value=dict(self.artifact)):
            with self.assertRaisesRegex(RuntimeError, "Docker unavailable"):
                await self.task.evaluate(self.archive)

    async def test_grader_is_fresh_strict_and_always_cleaned(self) -> None:
        """Use the manifest image, strict check/apply and one merged evaluation stream."""
        workspace = SimpleNamespace(start=AsyncMock(), close=AsyncMock(), copy_in=AsyncMock(),
                                    exec=AsyncMock(return_value=CommandResult(0, "official output", "")))
        with patch.object(swebench_task, "DockerWorkspace", return_value=workspace) as factory, \
                patch.object(swebench_task, "_check_base", new=AsyncMock()), \
                patch.object(swebench_task, "_check_syntax", new=AsyncMock(return_value={"errors": []})), \
                patch.object(swebench_task, "_checked", new=AsyncMock()) as checked:
            log = self.directory / "eval.log"
            await self.task._run_grader("source patch", log, self.artifact["changes"])
            self.assertEqual(factory.call_args.args[0], self.image)
            self.assertEqual(factory.call_args.kwargs["network"], "none")
            calls = [call.args[1] for call in checked.call_args_list]
            self.assertIn(["git", "-c", "safe.directory=/testbed", "apply", "--check",
                           "/workspace/submission.patch"], calls)
            self.assertEqual(workspace.exec.call_args.args[0], ["bash", "-c", "exec bash /workspace/eval.sh 2>&1"])
            self.assertEqual(log.read_text(), "official output")
            workspace.close.assert_awaited_once()
            workspace.exec.side_effect = RuntimeError("daemon")
            with self.assertRaisesRegex(RuntimeError, "daemon"):
                await self.task._run_grader("", log, self.artifact["changes"])
            self.assertEqual(workspace.close.await_count, 2)

    async def test_verified_candidate_syntax_error_scores_zero_without_official_report(self) -> None:
        """A healthy baseline plus explicit candidate compiler error is trainable zero."""
        syntax = {"baseline": {"errors": []}, "candidate": {"errors": [{"type": "IndentationError"}]}}
        self.task._run_grader = AsyncMock(return_value=syntax)
        with patch.object(swebench_task, "build_submission", return_value=dict(self.artifact)):
            reward = await self.task.evaluate(self.archive)
        self.assertEqual(reward.value, 0.)
        self.assertEqual(reward.metadata["status"], "syntax_error")
        self.assertTrue(reward.metadata["trainable"])
        self.assertEqual(reward.metadata["evaluated"], 0)
        self.grade.assert_not_called()
        evidence = json.loads(Path(reward.metadata["syntax_path"]).read_text(encoding="utf-8"))
        self.assertEqual(evidence["artifact_hash"], "content")

    async def test_baseline_syntax_failure_is_infrastructure(self) -> None:
        """A broken baseline must not be mistaken for model-generated syntax damage."""
        workspace = SimpleNamespace(start=AsyncMock(), close=AsyncMock(), copy_in=AsyncMock())
        with patch.object(swebench_task, "DockerWorkspace", return_value=workspace), \
                patch.object(swebench_task, "_check_base", new=AsyncMock()), \
                patch.object(swebench_task, "_checked", new=AsyncMock()), \
                patch.object(swebench_task, "_check_syntax", new=AsyncMock(return_value={"errors": ["bad"]})):
            with self.assertRaisesRegex(RuntimeError, "baseline source"):
                await self.task._run_grader("", self.directory / "log", self.artifact["changes"])
            workspace.close.assert_awaited_once()

    def test_evaluator_source_drift_is_rejected(self) -> None:
        """Same-version grading changes or added harness files cannot silently change scoring."""
        for hashes in ({"harness/grading.py": "changed"}, {"harness/grading.py": "fixed", "harness/new.py": "new"}, {}):
            with self.subTest(hashes=hashes), patch.object(
                    swebench_task, "evaluator_source_hashes", return_value=hashes):
                with self.assertRaisesRegex(RuntimeError, "evaluator source"):
                    swebench_task.SWEBenchTask(self.prompt, {"registry_path": str(self.registry)})

    def test_registry_and_baseline_identity_are_checked(self) -> None:
        """A modified private registry or trusted baseline cannot silently change the task."""
        self.prompt.ground_truth["registry_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "registry identity"):
            swebench_task.SWEBenchTask(self.prompt, {"registry_path": str(self.registry)})
        self.prompt.ground_truth["registry_sha256"] = hashlib.sha256(self.registry.read_bytes()).hexdigest()
        (self.directory / "base.tar").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "baseline archive"):
            swebench_task.SWEBenchTask(self.prompt, {"registry_path": str(self.registry)})
