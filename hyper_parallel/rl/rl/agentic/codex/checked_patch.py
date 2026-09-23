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
"""Run the pinned Codex patch helper and confirm its effect on candidate files."""

from __future__ import annotations

import difflib
import hashlib
from itertools import islice
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys


_PATCH_HEADERS = ("*** Update File: ", "*** Add File: ", "*** Delete File: ", "*** Move to: ")
# The pinned CLI receives the whole patch as one argv value; leave room below Linux MAX_ARG_STRLEN.
_MAX_PATCH_BYTES = 120 * 1024
_MAX_DIFF_LINES = 80
_MAX_DIFF_CHARS = 8000


def _patch_paths(patch: str) -> tuple[str, ...]:
    lines = patch.splitlines()
    if len(lines) < 3 or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise ValueError("Expected a complete Codex *** Begin Patch / *** End Patch block")
    paths = []
    for line in lines[1:-1]:
        for header in _PATCH_HEADERS:
            if line.startswith(header):
                name = line[len(header):]
                canonical = PurePosixPath(name)
                if (not name or name != name.strip() or name.startswith("/") or "\\" in name
                        or canonical.as_posix() != name or ".." in canonical.parts):
                    raise ValueError(f"Patch path must be canonical and relative to /workspace: {name!r}")
                paths.append(name)
                break
    if not paths:
        raise ValueError("Patch does not name a file to change")
    return tuple(dict.fromkeys(paths))


def _read_file(root: Path, name: str) -> bytes | None:
    target = root
    for part in PurePosixPath(name).parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f"Patch path contains a symlink: {name}")
    if not target.exists():
        return None
    if not stat.S_ISREG(target.stat().st_mode):
        raise ValueError(f"Patch path is not a regular file: {name}")
    return target.read_bytes()


def _report_change(name: str, before: bytes | None, after: bytes | None) -> None:
    old_hash = hashlib.sha256(before).hexdigest() if before is not None else "absent"
    new_hash = hashlib.sha256(after).hexdigest() if after is not None else "absent"
    print(f"VERIFIED {name}: sha256 {old_hash} -> {new_hash}")
    old_lines = (before or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
    new_lines = (after or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"before/{name}", tofile=f"after/{name}")
    excerpt = "".join(islice(diff, _MAX_DIFF_LINES + 1))
    if len(excerpt) > _MAX_DIFF_CHARS:
        excerpt = excerpt[:_MAX_DIFF_CHARS] + "\n... diff truncated; hashes above cover full files ...\n"
    elif excerpt.count("\n") > _MAX_DIFF_LINES:
        excerpt = ("\n".join(excerpt.splitlines()[:_MAX_DIFF_LINES])
                   + "\n... diff truncated; hashes above cover full files ...\n")
    if excerpt:
        print(excerpt, end="" if excerpt.endswith("\n") else "\n")


def apply_checked_patch(patch: str, root: Path, executable: Path) -> int:
    """Apply one Codex patch and verify actual bytes without using a Git index.

    Args:
        patch: Complete Codex patch block.
        root: Candidate source root, normally ``/workspace``.
        executable: Candidate-local ``apply_patch`` symlink to the pinned CLI.

    Returns:
        Zero only when the patch helper succeeds and at least one named file changes.
        Otherwise returns a nonzero process status and prints the real failure.
    """
    try:
        if len(patch.encode("utf-8")) > _MAX_PATCH_BYTES:
            raise ValueError("Patch exceeds the 120 KiB command limit")
        if not root.is_dir():
            raise ValueError(f"Candidate workspace is missing: {root}")
        paths = _patch_paths(patch)
        before = {name: _read_file(root, name) for name in paths}
    except (OSError, ValueError) as error:
        print(f"PATCH REJECTED: {error}", file=sys.stderr)
        return 2

    try:
        result = subprocess.run([str(executable), patch], cwd=root, capture_output=True,
                                text=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"PATCH COMMAND FAILED: {error}", file=sys.stderr)
        return 124 if isinstance(error, subprocess.TimeoutExpired) else 127
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")

    try:
        after = {name: _read_file(root, name) for name in paths}
    except (OSError, ValueError) as error:
        print(f"PATCH VERIFICATION FAILED: {error}", file=sys.stderr)
        return 2
    changed = [name for name in paths if before[name] != after[name]]
    syntax_errors = []
    for name in changed:
        _report_change(name, before[name], after[name])
        if name.endswith(".py") and after[name] is not None:
            try:
                compile(after[name], name, "exec", dont_inherit=True)
            except SyntaxError as error:
                print(f"PYTHON SYNTAX ERROR: {name}:{error.lineno}:{error.offset}: {error.msg}; "
                      "file remains changed", file=sys.stderr)
                syntax_errors.append(name)
    if result.returncode:
        print(f"PATCH FAILED: apply_patch exited {result.returncode}; changed files: {len(changed)}",
              file=sys.stderr)
        return result.returncode if 0 < result.returncode < 256 else 1
    if not changed:
        print("PATCH FAILED: apply_patch exited 0 but no named file changed", file=sys.stderr)
        return 1
    if syntax_errors:
        print(f"PATCH FAILED: {len(syntax_errors)} changed Python file(s) have syntax errors; "
              "fix the retained files before testing", file=sys.stderr)
        return 1
    print(f"PATCH VERIFIED: {len(changed)} file(s) changed; inspect the diff and run a focused test.")
    return 0


def main() -> int:
    """Read a patch from stdin inside the candidate's isolated Codex home."""
    return apply_checked_patch(sys.stdin.read(), Path("/workspace"), Path(__file__).with_name("apply_patch"))


if __name__ == "__main__":
    sys.exit(main())
