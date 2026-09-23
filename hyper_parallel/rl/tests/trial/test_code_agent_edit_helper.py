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
"""Candidate patch feedback must reflect source bytes rather than command status."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

_HELPER = Path(__file__).resolve().parents[2] / "rl/agentic/codex/checked_patch.py"
_SPEC = importlib.util.spec_from_file_location("checked_patch", _HELPER)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Candidate patch helper cannot be loaded")
checked_patch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checked_patch)
apply_checked_patch = checked_patch.apply_checked_patch


def _update(path: str = "src/_pytest/logging.py") -> str:
    """Build a minimal complete update block for a candidate source path."""
    return f"*** Begin Patch\n*** Update File: {path}\n@@\n-old\n+new\n*** End Patch\n"


class TestCheckedPatch(unittest.TestCase):
    """Verify the wrapper's observable command, path and byte contracts."""

    def setUp(self) -> None:
        """Create a source file and isolated capture streams for each case."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "src/_pytest/logging.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("old\n", encoding="utf-8")
        self.output = io.StringIO()
        self.error = io.StringIO()

    def _run(self, candidate: object, patch_text: str | None = None) -> int:
        """Invoke the helper with a controlled CLI result and capture its feedback."""
        with patch.object(checked_patch.subprocess, "run", side_effect=candidate) as command:
            with redirect_stdout(self.output), redirect_stderr(self.error):
                status = apply_checked_patch(patch_text or _update(), self.root, Path("/opt/alias/apply_patch"))
        if status != 2:
            self.assertEqual(command.call_args.args, (["/opt/alias/apply_patch", patch_text or _update()],))
            self.assertEqual(command.call_args.kwargs["cwd"], self.root)
            self.assertTrue(command.call_args.kwargs["capture_output"])
        return status

    def test_success_requires_readback_and_reports_diff(self) -> None:
        """A successful CLI call is accepted only after the named source changed."""
        def edit(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
            """Simulate a CLI command that writes the expected source bytes."""
            self.source.write_text("new\n", encoding="utf-8")
            return subprocess.CompletedProcess([], 0, "Success. Updated file.\n", "")

        self.assertEqual(self._run(edit), 0)
        self.assertIn("Success. Updated file.", self.output.getvalue())
        self.assertIn("-old", self.output.getvalue())
        self.assertIn("+new", self.output.getvalue())
        self.assertIn("PATCH VERIFIED: 1 file(s) changed", self.output.getvalue())
        self.assertEqual(self.error.getvalue(), "")

    def test_syntax_error_after_changed_bytes_fails_without_rollback(self) -> None:
        """A successful CLI write cannot report success for invalid retained Python."""
        self.source.write_text("def solve():\n    pass\n", encoding="utf-8")

        def invalid_edit(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
            """Model a fuzzy patch that dedents a Python function body."""
            self.source.write_text("def solve():\npass\n", encoding="utf-8")
            return subprocess.CompletedProcess([], 0, "Success. Updated file.\n", "")

        self.assertEqual(self._run(invalid_edit), 1)
        self.assertEqual(self.source.read_text(encoding="utf-8"), "def solve():\npass\n")
        self.assertIn("VERIFIED src/_pytest/logging.py", self.output.getvalue())
        self.assertNotIn("PATCH VERIFIED:", self.output.getvalue())
        self.assertIn("PYTHON SYNTAX ERROR: src/_pytest/logging.py:2:1:", self.error.getvalue())
        self.assertIn("file remains changed", self.error.getvalue())
        self.assertIn("fix the retained files", self.error.getvalue())

    def test_zero_exit_without_change_is_failure(self) -> None:
        """The historical sed zero-match failure cannot be reported as an edit."""
        candidate = lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "Success.\n", "")
        self.assertEqual(self._run(candidate), 1)
        self.assertEqual(self.source.read_text(encoding="utf-8"), "old\n")
        self.assertIn("exited 0 but no named file changed", self.error.getvalue())

    def test_cli_failure_preserves_real_status_and_partial_change_feedback(self) -> None:
        """A failed patch remains failed even if the CLI already changed one file."""
        def partial(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
            """Simulate a partially applied patch followed by a CLI error."""
            self.source.write_text("new\n", encoding="utf-8")
            return subprocess.CompletedProcess([], 7, "", "hunk not found\n")

        self.assertEqual(self._run(partial), 7)
        self.assertIn("hunk not found", self.error.getvalue())
        self.assertIn("apply_patch exited 7; changed files: 1", self.error.getvalue())
        self.assertIn("VERIFIED src/_pytest/logging.py", self.output.getvalue())

    def test_unmatched_hunk_keeps_real_error_and_no_change(self) -> None:
        """A failed context match must tell the candidate that nothing changed."""
        candidate = lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "Failed to find lines\n")
        self.assertEqual(self._run(candidate), 1)
        self.assertEqual(self.source.read_text(encoding="utf-8"), "old\n")
        self.assertIn("Failed to find lines", self.error.getvalue())
        self.assertIn("changed files: 0", self.error.getvalue())

    def test_added_file_is_verified_without_git(self) -> None:
        """New source files are measured from absent bytes to their written content."""
        added = self.root / "src/new.py"
        patch_text = "*** Begin Patch\n*** Add File: src/new.py\n+answer = 42\n*** End Patch\n"

        def add(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
            """Simulate a CLI command adding the named candidate source file."""
            added.write_text("answer = 42\n", encoding="utf-8")
            return subprocess.CompletedProcess([], 0, "Success. Added file.\n", "")

        self.assertEqual(self._run(add, patch_text), 0)
        self.assertIn("sha256 absent ->", self.output.getvalue())
        self.assertIn("+answer = 42", self.output.getvalue())

    def test_invalid_and_linked_paths_fail_before_cli(self) -> None:
        """Absolute, escaping and linked paths cannot be passed to the patch CLI."""
        for name in ("/tmp/other.py", "../other.py", "src/../other.py", "src//other.py"):
            with self.subTest(name=name):
                self.output.seek(0)
                self.output.truncate()
                self.error.seek(0)
                self.error.truncate()
                with patch.object(checked_patch.subprocess, "run") as command:
                    with redirect_stdout(self.output), redirect_stderr(self.error):
                        self.assertEqual(apply_checked_patch(_update(name), self.root, Path("/opt/alias")), 2)
                    command.assert_not_called()
                self.assertIn("canonical and relative", self.error.getvalue())

        link = self.root / "src/linked.py"
        link.symlink_to(self.source)
        with patch.object(checked_patch.subprocess, "run") as command:
            with redirect_stdout(self.output), redirect_stderr(self.error):
                self.assertEqual(apply_checked_patch(_update("src/linked.py"), self.root,
                                                     Path("/opt/alias")), 2)
            command.assert_not_called()
        self.assertIn("contains a symlink", self.error.getvalue())


if __name__ == "__main__":
    unittest.main()
