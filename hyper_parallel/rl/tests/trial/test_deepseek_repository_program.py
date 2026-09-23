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
"""Mock the DeepSeek candidate lifecycle around frozen repository grading."""

from __future__ import annotations

import asyncio
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from rl.agentic.core.types import RewardResult
from rl.agentic.ds_harness import harness


_IMAGE = "sha256:" + "a" * 64


def _budget_end(*, status: int = 409, kind: str = "error") -> list[dict[str, Any]]:
    return [{"type": "turn/end", "data": {"reason": {"kind": kind, "error": {
        "message": "DeepSeek repository reached max_completions=1",
        "code": f"HTTP_{status}", "status": status,
    }}}}]


def _run_async(test: Any) -> Any:
    @wraps(test)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Run the asynchronous contract in its own event loop."""
        return asyncio.run(test(*args, **kwargs))

    return wrapper


class _Workspace:
    def __init__(self, events: list[str]) -> None:
        self.name = "candidate"
        self.events = events
        self.async_stopped = asyncio.Event()
        self.stopped = False
        self.closed = False

    async def start(self) -> None:
        """Record startup for lifecycle ordering assertions."""
        self.events.append("start")

    async def copy_in(self, _source: Path, destination: str) -> None:
        """Verify the candidate home path and record installation."""
        assert destination == "/opt/hyper-codex-home"
        self.events.append("copy_in")

    async def exec(self, _argv: list[str], *, cwd: str) -> Any:
        """Record patch-helper installation and return a successful command result."""
        assert cwd == "/"
        self.events.append("install_patch")
        return SimpleNamespace(returncode=0)

    async def stop(self) -> None:
        """Mark the candidate stopped and unblock the simulated SDK."""
        self.events.append("stop")
        self.stopped = True
        self.async_stopped.set()

    async def export(self, destination: Path) -> Path:
        """Require the candidate to stop before exposing its archive."""
        assert self.stopped
        self.events.append("export")
        return destination

    async def close(self) -> None:
        """Record cleanup and propagate an injected cleanup failure if configured."""
        self.events.append("close")
        self.closed = True


class _Relay:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.budget_exhausted = asyncio.Event()
        self.budget_termination: dict[str, Any] | None = None
        self.close_error: Exception | None = None

    async def start(self) -> str:
        """Record startup for lifecycle ordering assertions."""
        self.events.append("relay_start")
        return "http://127.0.0.1:18080/v1"

    async def close(self) -> None:
        """Record cleanup and propagate an injected cleanup failure if configured."""
        self.events.append("relay_close")
        if self.close_error is not None:
            raise self.close_error


class _Task:
    workspace_image = _IMAGE

    def __init__(self, events: list[str], workspace: _Workspace) -> None:
        self.events = events
        self.workspace = workspace

    async def prepare(self, _workspace: _Workspace) -> None:
        """Record task preparation before candidate execution."""
        self.events.append("prepare")

    async def evaluate(self, archive: Path) -> RewardResult:
        """Require a frozen archive and return the controlled independent grade."""
        assert self.workspace.stopped and archive.name == "submission.tar"
        self.events.append("evaluate")
        return RewardResult(1.0, {"success": 1.0}, {"status": "resolved"})


def _program(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, _Workspace, _Relay, list[str]]:
    events: list[str] = []
    workspace = _Workspace(events)
    relay = _Relay(events)
    task = _Task(events, workspace)
    monkeypatch.setattr(harness, "DockerWorkspace", lambda **_kwargs: workspace)
    monkeypatch.setattr(harness, "CandidateRelay", lambda *_args, **_kwargs: relay)
    monkeypatch.setattr(harness, "load_reward_callable", lambda *_args: lambda *_args: task)
    rows: list[dict[str, Any]] = []

    def capture_rows(**kwargs: Any) -> tuple[Any, ...]:
        """Capture trajectory construction inputs for contract assertions."""
        rows.append(kwargs)
        return (SimpleNamespace(reward=kwargs["reward"], metadata=kwargs["metadata"]),)

    monkeypatch.setattr(harness, "build_harness_call_trajectories", capture_rows)

    async def immediate_thread(callable_: Any, *args: Any) -> Any:
        """Execute the controlled SDK stub within the test event loop."""
        return callable_(*args)

    monkeypatch.setattr(harness.asyncio, "to_thread", immediate_thread)
    prompt = SimpleNamespace(prompt_id="repository", messages=[SimpleNamespace(content="fix the code")])
    program = harness.DeepSeekAgentProgram(
        prompt, policy_version=4, sample_index=0,
        gateway_url="http://127.0.0.1:8300/v1", admin_url="http://127.0.0.1:8300",
        admin_token="controller-secret",
        config={"task_factory": "example:build_task", "workspace": {"image": _IMAGE},
                "max_turns": 1, "max_new_tokens": 128, "temperature": 0.7, "top_p": 0.8, "top_k": 20},
        end_of_turn_token_id=None,
    )
    program._admin_request = AsyncMock()
    program._run_harness = lambda *_args: ("done", "completed", [])
    program._captured_rows = rows
    return program, workspace, relay, events


@_run_async
async def test_natural_completion_freezes_before_independent_grading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The candidate stops and exports before the task receives its archive."""
    program, workspace, _relay, events = _program(monkeypatch, tmp_path)
    program._admin_request.return_value = {"policy_version": 4, "failure": None, "completions": [{"ordinal": 0}]}
    rows = await asyncio.wait_for(program._run_repository("session", tmp_path, 2), 5)
    assert rows[0].reward == 1.0
    assert events.index("stop") < events.index("export") < events.index("evaluate")
    assert workspace.closed


@_run_async
async def test_model_failure_keeps_zero_reward_without_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Verified bad model output remains trainable but cannot use residual edits."""
    program, workspace, _relay, events = _program(monkeypatch, tmp_path)
    program._admin_request.return_value = {
        "policy_version": 4, "completions": [{"ordinal": 0}],
        "failure": {"failure_origin": "model", "failure_reason": "invalid_json", "trainable": True},
    }
    rows = await program._run_repository("session", tmp_path, 2)
    assert rows[0].reward == 0.0
    assert "export" not in events and "evaluate" not in events
    assert workspace.closed


@_run_async
async def test_infrastructure_failure_rejects_episode_without_grading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A captured transport failure cannot become a zero reward sample."""
    program, workspace, _relay, events = _program(monkeypatch, tmp_path)
    program._admin_request.return_value = {
        "policy_version": 4, "completions": [{"ordinal": 0}],
        "failure": {"failure_origin": "infrastructure", "failure_reason": "relay failed", "trainable": False},
    }
    with pytest.raises(RuntimeError, match="not trainable"):
        await program._run_repository("session", tmp_path, 2)
    assert "export" not in events and "evaluate" not in events
    assert workspace.closed


@_run_async
async def test_budget_freezes_existing_edits_after_sdk_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Only a matching complete terminal budget may grade the frozen workspace."""
    program, workspace, relay, events = _program(monkeypatch, tmp_path)
    terminal = {"reason": "max_completions", "limit": 1, "completed": 1, "policy_version": 4}
    relay.budget_termination = terminal
    program._admin_request.return_value = {
        "policy_version": 4, "failure": None, "completions": [{"ordinal": 0}], "termination": terminal,
    }
    async def budget_thread(_callable: Any, *_args: Any) -> Any:
        """Deliver the budget signal and wait for the candidate to stop."""
        relay.budget_exhausted.set()
        await workspace.async_stopped.wait()
        return ("", "error", _budget_end())

    monkeypatch.setattr(harness.asyncio, "to_thread", budget_thread)
    rows = await program._run_repository("session", tmp_path, 2)
    assert rows[0].reward == 1.0
    assert rows[0].metadata["repository_budget_truncated"] is True
    assert events.index("stop") < events.index("export") < events.index("evaluate")
    assert workspace.closed


@pytest.mark.parametrize("mode", ["exception", "http500", "aborted"])
@_run_async
async def test_budget_rejects_unrelated_sdk_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str,
) -> None:
    """A Gateway 409 alone cannot launder another SDK ending into a scored patch."""
    program, workspace, relay, events = _program(monkeypatch, tmp_path)
    terminal = {"reason": "max_completions", "limit": 1, "completed": 1, "policy_version": 4}
    relay.budget_termination = terminal
    program._admin_request.return_value = {
        "policy_version": 4, "failure": None, "completions": [{"ordinal": 0}], "termination": terminal,
    }

    async def budget_thread(_callable: Any, *_args: Any) -> Any:
        """Deliver the budget signal and wait for the candidate to stop."""
        relay.budget_exhausted.set()
        await workspace.async_stopped.wait()
        if mode == "exception":
            raise RuntimeError("unrelated SDK exception")
        if mode == "http500":
            return ("", "error", _budget_end(status=500))
        return ("", "aborted", _budget_end())

    monkeypatch.setattr(harness.asyncio, "to_thread", budget_thread)
    expected = "unrelated SDK exception" if mode == "exception" else "matching SDK HTTP 409"
    with pytest.raises(RuntimeError, match=expected):
        await program._run_repository("session", tmp_path, 2)
    assert workspace.stopped and workspace.closed
    assert "export" not in events and "evaluate" not in events


@_run_async
async def test_relay_cleanup_failure_still_closes_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The container must be removed even when its model relay fails to close."""
    program, workspace, relay, events = _program(monkeypatch, tmp_path)
    relay.close_error = RuntimeError("relay close failed")
    with pytest.raises(RuntimeError, match="relay close failed"):
        await program._run_repository("session", tmp_path, 2)
    assert "close" in events and workspace.closed


@_run_async
async def test_cancellation_stops_sdk_before_removing_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Cancellation waits for the SDK's stopped subprocess before container cleanup."""
    program, workspace, _relay, events = _program(monkeypatch, tmp_path)
    sdk_started = asyncio.Event()

    async def blocked_thread(_callable: Any, *_args: Any) -> Any:
        """Keep the simulated SDK active until candidate cancellation stops it."""
        sdk_started.set()
        await workspace.async_stopped.wait()
        return ("", "aborted", [])

    monkeypatch.setattr(harness.asyncio, "to_thread", blocked_thread)
    operation = asyncio.create_task(program._run_repository("session", tmp_path, 2))
    await asyncio.wait_for(sdk_started.wait(), 2)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, 2)
    assert workspace.stopped and workspace.closed
    assert events.index("stop") < events.index("close")
    assert "export" not in events and "evaluate" not in events


def test_budget_rows_report_max_completions() -> None:
    """Budget-frozen calls report their actual terminal condition."""
    assert harness._trajectory_status([], {"repository_budget_truncated": True}) == (True, "max_completions")


def test_sdk_runtime_uses_candidate_paths_and_session_only_model_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The SDK starts Docker stdio with candidate paths and no controller secret."""
    program, _workspace, _relay, _events = _program(monkeypatch, tmp_path)
    configs: list[dict[str, Any]] = []

    class Config:
        def __init__(self, **kwargs: Any) -> None:
            configs.append(kwargs)

    class SDK:
        def __init__(self, _config: Any) -> None:
            pass

        def __enter__(self) -> SDK:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        @staticmethod
        def run(_instruction: str, *, session_id: str) -> Any:
            """Return a completed SDK result with the requested session identity."""
            return SimpleNamespace(session_id=session_id, final_response="done", finish_reason="completed",
                                   events=[], notifications=[])

    monkeypatch.setattr(harness, "_load_sdk", lambda _version: (SDK, Config))
    harness.DeepSeekAgentProgram._run_harness(
        program, "session", tmp_path, Path("/workspace"), tmp_path / "sessions",
        SimpleNamespace(name="candidate"), "http://127.0.0.1:18080/v1",
    )
    config = configs[0]
    argv = config["launch_args_override"]
    assert argv[:6] == ("docker", "exec", "-i", "-w", "/workspace", "-e")
    assert "DSH_CORDIS_CONFIG=/opt/deepseek-harness/cordis.yml" in argv
    assert "DEEPSEEK_BASE_URL=http://127.0.0.1:18080/v1" in argv
    assert "DEEPSEEK_API_KEY=session" in argv
    assert "controller-secret" not in " ".join(argv)
    assert config["cwd"] == "/workspace"
    assert config["runtime_cwd"] == str(tmp_path)
