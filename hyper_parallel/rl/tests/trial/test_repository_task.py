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
"""Frozen repository identity, independent scoring and failure contracts."""

import asyncio
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import pyarrow.parquet as pq

from rl.agentic.envs.docker_workspace import (
    CommandResult, InvalidSubmissionError, WorkspaceOutputLimitError, WorkspaceTimeoutError,
)
from rl.dataset.contracts import Message, PromptRecord
from examples.code_agent.prepare_data import adapt_row, prepare_data
from examples.code_agent.task import RepositoryTask, content_hash, load_fixture, task_manifest, validate_submission


def _task(fixture_id: str = "merge_intervals", **config: object) -> RepositoryTask:
    prompt = PromptRecord("test", (Message("user", "repair"),), task_manifest(fixture_id),
                          {"task_type": "repository_python_cli"})
    return RepositoryTask(prompt, {"image": "sha256:" + "1" * 64, **config})


def _archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def _grader(result: object) -> SimpleNamespace:
    return SimpleNamespace(start=AsyncMock(), copy_in=AsyncMock(), close=AsyncMock(),
                           exec=AsyncMock(side_effect=[CommandResult(0, '{"runtime": "ready"}\n', ""), result]))


class TestRepositorySubmission(unittest.TestCase):
    """Check source change bounds and public task identity without running code."""

    def test_changes_include_add_modify_delete(self) -> None:
        """The reference exercises all three operations without mutating trusted baseline."""
        fixture, base = load_fixture("word_counts")
        files = dict(base)
        for name, content in fixture["reference"].items():
            if content is None:
                del files[name]
            else:
                files[name] = content.encode()
        artifact = validate_submission(base, files)
        self.assertEqual(artifact["changes"], {"added": ["src/normalization.py"],
                                            "modified": ["src/helpers.py", "src/solution.py"],
                                            "deleted": ["src/legacy.py"]})
        self.assertNotEqual(artifact["content_hash"], content_hash(base))
        self.assertNotIn("src/normalization.py", base)
        self.assertEqual(validate_submission(base, base)["changes"], {"added": [], "modified": [], "deleted": []})

    def test_protected_and_noncanonical_paths_rejected(self) -> None:
        """Only direct canonical src Python files can change."""
        _, base = load_fixture("word_counts")
        for name in ("public_test.py", "entrypoint.py", ".git/config", "../escape.py", "src/link",
                     "src/nested/solution.py", "src//new.py", "src/../new.py", "src/new\\file.py"):
            with self.subTest(name=name), self.assertRaises(InvalidSubmissionError):
                validate_submission(base, {**base, name: b"tampered"})
        files = dict(base)
        del files["entrypoint.py"]
        with self.assertRaises(InvalidSubmissionError):
            validate_submission(base, files)

    def test_dataset_contains_only_public_problem_and_hashes(self) -> None:
        """Prepared data omits reference sources and private inputs/expectations."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tasks.parquet"
            prepare_data(path)
            rows = pq.read_table(path).to_pylist()
        self.assertEqual(len(rows), 2)
        for row in rows:
            prompt = adapt_row(row, 0)
            self.assertEqual(set(prompt.ground_truth), {"fixture_id", "base_hash", "test_version"})
            self.assertEqual(prompt.metadata["task_type"], "repository_python_cli")
            self.assertNotIn("cases", row)
            self.assertNotIn("reference", row)
        rows[0]["ground_truth"]["base_hash"] = "tampered"
        with self.assertRaisesRegex(ValueError, "manifest"):
            adapt_row(rows[0], 0)

    def test_image_identity_and_budgets_are_explicit(self) -> None:
        """Mutable image tags and unbounded grader settings fail before execution."""
        for config in ({"image": "latest"}, {"run_timeout": float("nan")}, {"output_limit_bytes": 0},
                       {"python_executable": "python"}, {"unexpected": 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                _task(**config)


class TestRepositoryScoring(unittest.IsolatedAsyncioTestCase):
    """Check binary reward, fresh grader ownership and explicit infrastructure failures."""

    async def test_prepare_copies_only_frozen_public_bytes(self) -> None:
        """Workspace receives baseline files but never registry or reference patches."""
        task = _task()
        captured = {}

        async def capture(directory: Path) -> None:
            """Read the exact public files copied into the workspace."""
            captured.update({path.relative_to(directory).as_posix(): path.read_bytes()
                             for path in directory.rglob("*") if path.is_file()})

        await task.prepare(SimpleNamespace(copy_in=AsyncMock(side_effect=capture)))
        self.assertEqual(captured, task.base)
        self.assertNotIn("tasks.json", captured)
        self.assertNotIn("reference", captured)

    async def test_all_cases_use_new_graders_and_only_all_correct_scores(self) -> None:
        """Candidate success reports are not trusted and a later correct case cannot erase failure."""
        task = _task()
        for fake_success in (False, True):
            graders = [_grader(CommandResult(0, json.dumps(expected), "")) for _, expected in task.fixture["cases"]]
            if fake_success:
                graders[0] = _grader(CommandResult(0, '{"passed": true}', ""))
            with tempfile.TemporaryDirectory() as temp:
                archive = _archive(Path(temp) / "submission.tar", task.base)
                with patch("examples.code_agent.task.DockerWorkspace", side_effect=graders) as backend:
                    reward = await task.evaluate(archive)
                self.assertTrue(archive.with_suffix(".manifest.json").is_file())
            self.assertEqual(reward.value, 0 if fake_success else 1)
            self.assertEqual(reward.metadata["evaluated"], 5)
            self.assertEqual(backend.call_count, 5)
            for grader in graders:
                grader.close.assert_awaited_once()
                self.assertEqual(grader.exec.await_count, 2)
            for call in backend.call_args_list:
                self.assertEqual(call.kwargs["network"], "none")

    async def test_invalid_artifact_never_starts_grader(self) -> None:
        """Forbidden source changes are candidate failures; missing archives are infrastructure failures."""
        task = _task()
        with tempfile.TemporaryDirectory() as temp:
            archive = _archive(Path(temp) / "submission.tar", {**task.base, "public_test.py": b"print('PASS')"})
            with patch("examples.code_agent.task.DockerWorkspace") as backend:
                reward = await task.evaluate(archive)
                self.assertEqual(reward.metadata["status"], "invalid_submission")
                backend.assert_not_called()
                with self.assertRaises(FileNotFoundError):
                    await task.evaluate(Path(temp) / "absent.tar")

    async def test_candidate_limits_continue_to_later_infrastructure_failure(self) -> None:
        """Bounded candidate failures score zero unless a later infrastructure error invalidates scoring."""
        task = _task()
        for error_type in (WorkspaceTimeoutError, WorkspaceOutputLimitError):
            for infrastructure_failure in (False, True):
                graders = [_grader(CommandResult(0, json.dumps(expected), ""))
                           for _, expected in task.fixture["cases"]]
                graders[0] = _grader(error_type(CommandResult(-1, "", "")))
                if infrastructure_failure:
                    graders[1] = _grader(RuntimeError("Docker failed"))
                with tempfile.TemporaryDirectory() as temp:
                    archive = _archive(Path(temp) / "submission.tar", task.base)
                    with patch("examples.code_agent.task.DockerWorkspace", side_effect=graders):
                        if infrastructure_failure:
                            with self.assertRaisesRegex(RuntimeError, "Docker failed"):
                                await task.evaluate(archive)
                        else:
                            reward = await task.evaluate(archive)
                            self.assertEqual(reward.value, 0)
                            self.assertEqual(reward.metadata["evaluated"], 5)
                graders[0].close.assert_awaited_once()
                graders[1].close.assert_awaited_once()

    async def test_unhealthy_runtime_and_cancel_propagate_after_cleanup(self) -> None:
        """Missing interpreter and cancellation are never scored as candidate wrong answers."""
        task = _task()
        for result in (CommandResult(127, "", "missing interpreter"), asyncio.CancelledError()):
            grader = _grader(CommandResult(0, "[]", ""))
            grader.exec.side_effect = [result]
            with tempfile.TemporaryDirectory() as temp:
                archive = _archive(Path(temp) / "submission.tar", task.base)
                with patch("examples.code_agent.task.DockerWorkspace", return_value=grader):
                    error = asyncio.CancelledError if isinstance(result, asyncio.CancelledError) else RuntimeError
                    with self.assertRaises(error):
                        await task.evaluate(archive)
            grader.close.assert_awaited_once()
            grader.copy_in.assert_not_awaited()

    async def test_cleanup_failure_preserves_original_error(self) -> None:
        """A cleanup problem is visible without replacing the original infrastructure failure."""
        task = _task()
        grader = _grader(CommandResult(0, "[]", ""))
        grader.start.side_effect = RuntimeError("original start failure")
        grader.close.side_effect = RuntimeError("cleanup failure")
        with tempfile.TemporaryDirectory() as temp:
            archive = _archive(Path(temp) / "submission.tar", task.base)
            with patch("examples.code_agent.task.DockerWorkspace", return_value=grader), \
                    self.assertLogs("examples.code_agent.task", "ERROR"):
                with self.assertRaisesRegex(RuntimeError, "original start failure"):
                    await task.evaluate(archive)
