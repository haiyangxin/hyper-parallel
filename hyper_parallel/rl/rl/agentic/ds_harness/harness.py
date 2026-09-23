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
"""DeepSeek Harness SDK process boundary and Hyper-RL AgentProgram."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rl.agentic.core.program_runner import (
    HarnessProgramFactory,
    HarnessRuntime,
    build_harness_trajectory,
    build_harness_call_trajectories,
    harness_generation_settings,
    load_reward_callable,
    request_gateway_json,
)
from rl.agentic.core.types import RewardResult
from rl.agentic.envs.docker_workspace import DockerWorkspace, InvalidSubmissionError, acquire_workspace_slot
from rl.agentic.envs.model_relay import CandidateRelay
from rl.agentic.ds_harness.gateway import DeepSeekGateway
from rl.dataset.contracts import PromptRecord, Trajectory


_LOGGER = logging.getLogger(__name__)
DEFAULT_DEEPSEEK_HARNESS_VERSION = "0.1.1rc1"
RewardCallable = Callable[[str, PromptRecord], float | RewardResult]
_MAX_GENERATION_PREFIX_REWRITE = 16
_REPOSITORY_TOOL_INSTRUCTION = (
    "You are repairing the repository in /workspace. Use the declared bash tool for reading, editing, and "
    "testing; its arguments are command and description. There is no separate apply_patch model tool. "
    "To apply a patch, invoke bash with a command containing "
    "python /opt/hyper-codex-home/checked_patch.py and a shell heredoc with a complete "
    "*** Begin Patch / *** End Patch block. DS bash prints [exit code: N] for nonzero exits; a zero exit "
    "may have no exit marker. Inspect changed-file hashes and diff to confirm an edit. "
    "Read source in bounded chunks so the relevant lines remain visible. After editing, reread the "
    "changed lines and run a focused public test. A successful shell exit without a file change is not a fix. "
    "Do not use the skill tool for this repository task.\n\n"
)


def _trajectory_status(
    traces: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> tuple[bool, str]:
    """Infer completion status and stop reason from captured traces."""
    if metadata.get("repository_budget_truncated") is True:
        return True, "max_completions"
    finish_reasons = {str(trace.get("finish_reason") or "") for trace in traces}
    if finish_reasons & {"length", "max_tokens"}:
        return True, "max_tokens"
    harness_reason = str(metadata.get("finish_reason") or "")
    if harness_reason in {"error", "aborted"}:
        return True, f"harness_{harness_reason}"
    return False, "completed"


def build_deepseek_trajectory(
    *,
    prompt: PromptRecord,
    policy_version: int,
    sample_index: int,
    completion_records: Sequence[Mapping[str, Any]],
    reward: float,
    reward_components: Mapping[str, float],
    end_of_turn_token_id: int | None = None,
    max_episode_tokens: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Trajectory:
    """Merge every captured DeepSeek completion into one token-exact trajectory."""
    return build_harness_trajectory(
        label="DeepSeek",
        runner_name="deepseek",
        prompt=prompt,
        policy_version=policy_version,
        sample_index=sample_index,
        completion_records=completion_records,
        reward=reward,
        reward_components=reward_components,
        end_of_turn_token_id=end_of_turn_token_id,
        max_episode_tokens=max_episode_tokens,
        metadata=metadata,
        max_prefix_rewrite=_MAX_GENERATION_PREFIX_REWRITE,
        tool_history_field="messages",
        status_resolver=_trajectory_status,
    )


def _validated_failure(captured: Mapping[str, Any], finish_reason: str) -> dict[str, Any]:
    """Admit model failures only when the gateway explicitly established their origin."""
    failure = captured.get("failure")
    if failure is not None:
        if (not isinstance(failure, Mapping) or failure.get("failure_origin") != "model"
                or failure.get("trainable") is not True or failure.get("failure_reason") not in (
                    "call_budget_exhausted", "invalid_json", "invalid_tool_schema",
                )):
            raise RuntimeError(f"DeepSeek gateway failure is not trainable: {failure!r}")
        return dict(failure)
    if finish_reason in {"error", "aborted"}:
        raise RuntimeError(f"DeepSeek Harness {finish_reason} has unknown failure origin")
    return {}


def _confirmed_repository_budget_end(finish_reason: str | None, events: Sequence[Mapping[str, Any]],
                                     termination: Mapping[str, Any]) -> bool:
    """Accept only the SDK's observed terminal error for this exact Gateway 409."""
    if finish_reason != "error":
        return False
    expected = f"DeepSeek repository reached max_completions={termination['limit']}"
    for event in events:
        if event.get("type") != "turn/end":
            continue
        data = event.get("data")
        reason = data.get("reason") if isinstance(data, Mapping) else None
        error = reason.get("error") if isinstance(reason, Mapping) else None
        if (isinstance(error, Mapping) and reason.get("kind") == "error"
                and error.get("status") == 409 and error.get("code") == "HTTP_409"
                and error.get("message") == expected):
            return True
    return False


def _load_reward_callable(value: Any) -> RewardCallable:
    return load_reward_callable(value, "agentic.deepseek.reward_callable", "DeepSeek")


def _load_sdk(expected_version: str) -> tuple[Any, Any]:
    """Load the optional SDK only when the DeepSeek runner is selected."""
    try:
        installed = importlib.metadata.version("deepseek-harness-sdk")
        sdk = importlib.import_module("deepseek_harness")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError(
            f"The DeepSeek runner requires deepseek-harness-sdk=={expected_version}"
        ) from error
    if installed != expected_version:
        raise RuntimeError(
            "DeepSeek Harness SDK version mismatch: "
            f"expected {expected_version}, got {installed}"
        )
    return sdk.DeepSeekHarness, sdk.DeepSeekHarnessConfig


def _http_json(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None,
    timeout: float,
    admin_token: str | None = None,
    *, max_response_bytes: int = 32 * 1024 * 1024,
) -> dict[str, Any]:
    return request_gateway_json("DeepSeek", method, url, payload, timeout,
                                admin_token=admin_token, max_response_bytes=max_response_bytes)


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class DeepSeekRuntime(HarnessRuntime):
    """Own one DeepSeek-only gateway while Hyper-RL continues to own vLLM."""

    def __init__(self, engine: Any, config: Mapping[str, Any]) -> None:
        """Bind the independent runtime to the existing shared rollout engine."""
        super().__init__(engine, config, DeepSeekGateway, "DeepSeek", 8300, "/v1")


class DeepSeekAgentProgram:
    """Run one DeepSeek Harness episode and retain each real model-call context."""

    def __init__(
        self,
        prompt: PromptRecord,
        policy_version: int,
        sample_index: int,
        gateway_url: str,
        admin_url: str,
        config: Mapping[str, Any],
        end_of_turn_token_id: int | None,
        admin_token: str | None = None,
    ) -> None:
        """Capture the episode identity and immutable Harness settings."""
        self.prompt = prompt
        self.policy_version = policy_version
        self.sample_index = sample_index
        self.gateway_url = gateway_url.rstrip("/")
        self.admin_url = admin_url.rstrip("/")
        self.admin_token = admin_token
        self.config = dict(config)
        self.end_of_turn_token_id = end_of_turn_token_id
        task_factory = self.config.get("task_factory")
        if bool(task_factory) == bool(self.config.get("reward_callable")):
            raise ValueError("DeepSeek requires exactly one of task_factory and reward_callable")
        if task_factory and (not admin_url or not admin_token):
            raise ValueError("Repository DeepSeek requires controller admin_url and admin_token")
        self.task_factory = (load_reward_callable(task_factory, "agentic.deepseek.task_factory", "DeepSeek task")
                             if task_factory else None)
        self.reward_callable = None if task_factory else _load_reward_callable(self.config.get("reward_callable"))

    async def run(self) -> tuple[Trajectory, ...]:
        """Run the SDK, fetch exact network evidence, score, and convert it."""
        session_id = uuid.uuid4().hex
        artifact_dir, workspace_dir, session_root = self._prepare_directories(
            session_id
        )
        timeout = float(self.config.get("request_timeout", 600.0))
        try:
            await self._admin_request(
                "POST",
                "/internal/sessions",
                {
                    "session_id": session_id,
                    "policy_version": self.policy_version,
                    "artifact_dir": str(artifact_dir),
                    "max_completions": int(self.config["max_turns"]),
                    "generation": self._generation_settings(),
                    "repository_task": self.task_factory is not None,
                },
                timeout,
            )
            if self.task_factory is not None:
                capacity = int(self.config["workspace"].get("max_concurrent", 1))
                async with acquire_workspace_slot(artifact_dir.parent / ".workspace-slots", capacity, timeout):
                    return await self._run_repository(session_id, artifact_dir, timeout)
            started = time.perf_counter()
            final_answer, finish_reason, events = await asyncio.to_thread(
                self._run_harness,
                session_id,
                artifact_dir,
                workspace_dir,
                session_root,
            )
            captured = await self._admin_request(
                "GET",
                f"/internal/sessions/{session_id}",
                None,
                timeout,
            )
            reward_result = self._score_capture(final_answer, finish_reason, captured)
            return build_harness_call_trajectories(
                label="DeepSeek",
                runner_name="deepseek",
                tool_history_field="messages",
                status_resolver=_trajectory_status,
                prompt=self.prompt,
                policy_version=self.policy_version,
                sample_index=self.sample_index,
                completion_records=captured.get("completions", []),
                reward=reward_result.value,
                reward_components=reward_result.components,
                max_episode_tokens=(
                    None
                    if self.config.get("max_episode_tokens") is None
                    else int(self.config["max_episode_tokens"])
                ),
                metadata={
                    "deepseek_harness_version": str(
                        self.config.get("version", DEFAULT_DEEPSEEK_HARNESS_VERSION)
                    ),
                    "deepseek_session_id": session_id,
                    "artifact_dir": str(artifact_dir),
                    "workspace_dir": str(workspace_dir),
                    "generation_seconds": time.perf_counter() - started,
                    "final_answer": final_answer,
                    "finish_reason": finish_reason,
                    "harness_finish_reason": finish_reason,
                    "deepseek_events": events,
                    **dict(reward_result.metadata),
                },
            )
        finally:
            original_error = sys.exc_info()[1]
            try:
                await self._admin_request("DELETE", f"/internal/sessions/{session_id}", None, timeout)
            except Exception:
                if original_error is None:
                    raise
                _LOGGER.exception("DeepSeek session cleanup failed while preserving original failure")

    async def _run_repository(self, session_id: str, artifact_dir: Path,
                              timeout: float) -> tuple[Trajectory, ...]:
        """Run DS tools inside the candidate, then grade a stopped snapshot."""
        settings = dict(self.config["workspace"])
        settings.pop("max_concurrent", None)
        settings.pop("output_limit_bytes", None)
        run_id = settings.pop("run_id", self.config.get("run_id", session_id))
        task_config = dict(self.config.get("task_config", {}))
        if task_config.get("image", settings["image"]) != settings["image"]:
            raise ValueError("Repository candidate and grader must use the same pinned image")
        task_config["image"] = settings["image"]
        task_config["run_id"] = run_id
        task = self.task_factory(self.prompt, task_config)
        image = getattr(task, "workspace_image", settings["image"])
        if not isinstance(image, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
            raise ValueError("Repository task workspace_image must be a fixed sha256 image ID")
        settings["image"] = image
        workspace = DockerWorkspace(**settings, network="none", name_prefix="hyper-deepseek-candidate",
                                    run_id=run_id)
        relay = CandidateRelay(workspace, self.gateway_url, session_id, timeout,
                               int(self.config.get("max_request_bytes", 8 * 1024 * 1024)),
                               int(self.config.get("max_response_bytes", 32 * 1024 * 1024)),
                               admin_url=self.admin_url, admin_token=self.admin_token,
                               route="chat_completions")
        started = time.perf_counter()
        try:
            await workspace.start()
            await task.prepare(workspace)
            model_url = await relay.start()
            dsh_home = self._write_dsh_settings(artifact_dir)
            shutil.copyfile(Path(__file__).parents[1] / "codex" / "checked_patch.py",
                            dsh_home / "checked_patch.py")
            await workspace.copy_in(dsh_home, "/opt/hyper-codex-home")
            alias = await workspace.exec(["ln", "-s", "/opt/codex/bin/codex",
                                          "/opt/hyper-codex-home/apply_patch"], cwd="/")
            if alias.returncode:
                raise RuntimeError("Cannot install the candidate's checked patch command")
            final_answer, finish_reason, events = "", None, []
            execution_error = None
            budget_termination = None
            execution = asyncio.create_task(asyncio.to_thread(
                self._run_harness, session_id, artifact_dir, Path("/workspace"),
                artifact_dir / "sessions", workspace, model_url,
            ))
            budget = asyncio.create_task(relay.budget_exhausted.wait())
            try:
                await asyncio.wait((execution, budget), return_when=asyncio.FIRST_COMPLETED)
                if budget.done() and relay.budget_exhausted.is_set():
                    budget_termination = relay.budget_termination
                    await relay.close()
                    await workspace.stop()
                try:
                    final_answer, finish_reason, events = await execution
                except Exception as error:
                    execution_error = error
            finally:
                budget.cancel()
                await asyncio.gather(budget, return_exceptions=True)
                if not execution.done():
                    await workspace.stop()
                    await asyncio.shield(execution)
            await relay.close()
            if relay.budget_exhausted.is_set():
                budget_termination = relay.budget_termination
            captured = await self._admin_request("GET", f"/internal/sessions/{session_id}", None, timeout)
            if captured.get("policy_version") != self.policy_version:
                raise RuntimeError("DeepSeek gateway returned a different policy version")
            failure = captured.get("failure")
            if failure is not None and (not isinstance(failure, Mapping) or failure.get("failure_origin") != "model"
                                        or failure.get("trainable") is not True):
                raise RuntimeError(f"DeepSeek gateway failure is not trainable: {failure!r}") from execution_error
            if budget_termination is not None:
                if (failure is not None or captured.get("termination") != budget_termination
                        or budget_termination.get("policy_version") != self.policy_version
                        or budget_termination.get("limit") != int(self.config["max_turns"])
                        or len(captured.get("completions", [])) != budget_termination["limit"]):
                    raise RuntimeError("Repository budget termination differs from the captured session")
                if execution_error is not None:
                    raise RuntimeError("Repository budget ended with an unrelated SDK exception") from execution_error
                if not _confirmed_repository_budget_end(finish_reason, events, budget_termination):
                    raise RuntimeError("Repository budget lacks the matching SDK HTTP 409 terminal event")
            if execution_error is not None and failure is None and budget_termination is None:
                raise execution_error
            if finish_reason in {"error", "aborted"} and failure is None and budget_termination is None:
                raise RuntimeError(f"DeepSeek Harness ended with unknown-origin {finish_reason}")
            await workspace.stop()
            if failure is not None:
                reward = RewardResult(0.0, {"success": 0.0}, dict(failure))
            else:
                try:
                    archive = await workspace.export(artifact_dir / "submission.tar")
                except InvalidSubmissionError as error:
                    reward = RewardResult(0.0, {"success": 0.0},
                                          {"status": "invalid_submission", "reason": str(error)})
                else:
                    reward = await task.evaluate(archive)
                if not isinstance(reward, RewardResult):
                    raise TypeError("Repository task.evaluate must return RewardResult")
            if budget_termination is not None:
                reward = RewardResult(reward.value, reward.components,
                                      {**reward.metadata, "budget_termination": budget_termination,
                                       "repository_budget_truncated": True})
            return build_harness_call_trajectories(
                label="DeepSeek", runner_name="deepseek", tool_history_field="messages",
                status_resolver=_trajectory_status, prompt=self.prompt,
                policy_version=self.policy_version, sample_index=self.sample_index,
                completion_records=captured.get("completions", []), reward=reward.value,
                reward_components=reward.components,
                max_episode_tokens=self.config.get("max_episode_tokens"),
                metadata={"deepseek_harness_version": str(self.config.get("version", DEFAULT_DEEPSEEK_HARNESS_VERSION)),
                          "deepseek_session_id": session_id, "artifact_dir": str(artifact_dir),
                          "workspace_dir": "/workspace", "generation_seconds": time.perf_counter() - started,
                          "final_answer": final_answer, "finish_reason": finish_reason,
                          "harness_finish_reason": finish_reason, "deepseek_events": events,
                          **dict(reward.metadata)},
            )
        finally:
            original_error = sys.exc_info()[1]
            cleanup = asyncio.create_task(self._close_repository(relay, workspace))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await asyncio.gather(cleanup, return_exceptions=True)
                raise
            except Exception:
                if original_error is None:
                    raise
                _LOGGER.exception("DeepSeek relay cleanup failed while preserving original error")

    @staticmethod
    async def _close_repository(relay: CandidateRelay, workspace: DockerWorkspace) -> None:
        try:
            await relay.close()
        finally:
            await workspace.close()

    async def _admin_request(self, method: str, path: str, payload: Mapping | None,
                             timeout: float) -> dict[str, Any]:
        """Settle registration before cleanup when controller cancellation arrives."""
        request = asyncio.create_task(asyncio.to_thread(
            _http_json, method, self.admin_url + path, payload, timeout, self.admin_token,
            max_response_bytes=int(self.config.get("max_response_bytes", 32 * 1024 * 1024)),
        ))
        try:
            return await asyncio.shield(request)
        except asyncio.CancelledError:
            await asyncio.gather(request, return_exceptions=True)
            raise

    def _score_capture(
        self, final_answer: str, finish_reason: str, captured: Mapping[str, Any],
    ) -> RewardResult:
        """Validate failure provenance and score captured tool-call evidence."""
        contract_error = self._capture_contract_error(captured)
        failure = _validated_failure(captured, finish_reason)
        if failure:
            contract_error = f"DeepSeek model failure: {failure['failure_reason']}; {contract_error or ''}"
        reward_result = self.reward_callable(final_answer, self.prompt)
        if not isinstance(reward_result, RewardResult):
            reward_value = float(reward_result)
            reward_result = RewardResult(reward_value, {"outcome": reward_value})
        if finish_reason in {"error", "aborted"}:
            contract_error = (
                f"DeepSeek Harness ended with {finish_reason}"
                if contract_error is None
                else f"{contract_error}; Harness ended with {finish_reason}"
            )
        if contract_error is not None:
            reward_result = RewardResult(
                0.0,
                {**reward_result.components, "tool_contract": 0.0},
                {**reward_result.metadata, "tool_contract_error": contract_error},
            )
        elif self.config.get("required_tool_calls") is not None:
            reward_result = RewardResult(
                reward_result.value,
                {**reward_result.components, "tool_contract": 1.0},
                reward_result.metadata,
            )
        return RewardResult(
            reward_result.value, reward_result.components,
            {**failure, **dict(reward_result.metadata)},
        )

    def _validate_capture(self, captured: Mapping[str, Any]) -> None:
        """Validate captured policy identity and required tool-call evidence."""
        if captured.get("policy_version") != self.policy_version:
            raise RuntimeError("DeepSeek gateway returned a different policy version")
        expected_tool_calls = self.config.get("required_tool_calls")
        expected_tool_name = self.config.get("required_tool_name")
        if expected_tool_calls is None and expected_tool_name is None:
            return
        actual_tool_calls, tool_names = _captured_tool_calls(captured)
        if (
            expected_tool_calls is not None
            and actual_tool_calls != int(expected_tool_calls)
        ):
            raise RuntimeError(
                "DeepSeek episode violated its tool-call contract: "
                f"expected={int(expected_tool_calls)}, actual={actual_tool_calls}"
            )
        if expected_tool_name is not None and any(
            name != expected_tool_name for name in tool_names
        ):
            raise RuntimeError(
                "DeepSeek episode called an unexpected tool: "
                f"expected={expected_tool_name!r}, actual={tool_names}"
            )

    def _capture_contract_error(self, captured: Mapping[str, Any]) -> str | None:
        """Keep stochastic tool mistakes as zero-reward RL evidence."""
        try:
            self._validate_capture(captured)
        except RuntimeError as error:
            message = str(error)
            if "tool-call contract" not in message and "unexpected tool" not in message:
                raise
            return message
        return None

    def _generation_settings(self) -> dict[str, Any]:
        return harness_generation_settings(
            self.config,
            self.prompt.prompt_id,
            self.policy_version,
            self.sample_index,
            reasoning_effort=str(self.config.get("reasoning_effort", "off")),
        )

    def _prepare_directories(self, session_id: str) -> tuple[Path, Path, Path]:
        """Create isolated workspace and runtime directories for this session."""
        root = (
            Path(str(self.config.get("session_root", "/tmp/hyper-rl-deepseek")))
            .expanduser()
            .resolve()
        )
        artifact_dir = (
            root / f"{self.prompt.prompt_id}-{self.sample_index}-{session_id}"
        )
        workspace_dir = artifact_dir / "workspace"
        session_root = artifact_dir / "sessions"
        artifact_dir.mkdir(parents=True, exist_ok=False)
        template_value = self.config.get("workspace_template")
        if template_value:
            template = Path(str(template_value)).expanduser().resolve()
            if not template.is_dir():
                raise ValueError(
                    f"DeepSeek workspace template does not exist: {template}"
                )
            shutil.copytree(template, workspace_dir)
        else:
            workspace_dir.mkdir()
        session_root.mkdir()
        return artifact_dir, workspace_dir, session_root

    def _write_dsh_settings(self, artifact_dir: Path) -> Path:
        """Pin per-episode model behavior without reading ambient user settings."""
        dsh_home = artifact_dir / "dsh-home"
        dsh_home.mkdir()
        settings = {
            "agent-default-model": {
                "provider": str(self.config.get("provider", "deepseek-official")),
                "model": str(self.config.get("model", "policy")),
                "reasoningEffort": str(
                    self.config.get("reasoning_effort", "off")
                ),
            }
        }
        (dsh_home / "settings.yaml").write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return dsh_home

    def _run_harness(
        self,
        session_id: str,
        artifact_dir: Path,
        workspace_dir: Path,
        session_root: Path,
        workspace: DockerWorkspace | None = None,
        model_url: str | None = None,
    ) -> tuple[str, str | None, list[dict[str, Any]]]:
        """Run the external harness and retain its output for trajectory validation."""
        expected = str(self.config.get("version", DEFAULT_DEEPSEEK_HARNESS_VERSION))
        harness_class, config_class = _load_sdk(expected)
        dsh_home = self._write_dsh_settings(artifact_dir) if workspace is None else artifact_dir / "dsh-home"
        gateway_host = urlparse(self.gateway_url).hostname
        no_proxy = [os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))]
        no_proxy.extend(("127.0.0.1", "localhost"))
        if gateway_host:
            no_proxy.append(gateway_host)
        environment = {
            "DSH_HOME": str(dsh_home),
            "DSH_TELEMETRY_DISABLED": "1",
            "NO_PROXY": ",".join(filter(None, no_proxy)),
            "no_proxy": ",".join(filter(None, no_proxy)),
        }
        runtime_bin = self.config.get("runtime_bin")
        launch_args = None
        if workspace is not None:
            if model_url is None:
                raise ValueError("Repository DeepSeek requires the candidate model relay")
            launch_args = ("docker", "exec", "-i", "-w", "/workspace",
                           "-e", "DSH_HOME=/opt/hyper-codex-home",
                           "-e", "DSH_CWD=/workspace",
                           "-e", f"DSH_SESSION_ROOT=/tmp/hyper-dsh-{session_id}",
                           "-e", "DSH_CORDIS_CONFIG=/opt/deepseek-harness/cordis.yml",
                           "-e", "DSH_TELEMETRY_DISABLED=1",
                           "-e", f"DEEPSEEK_BASE_URL={model_url}",
                           "-e", f"DEEPSEEK_API_KEY={session_id}",
                           "-e", "NO_PROXY=127.0.0.1,localhost",
                           "-e", "no_proxy=127.0.0.1,localhost",
                           workspace.name, "/opt/deepseek-harness/runtime/dsh-jsonrpc-agent-pkg-linux-arm64")
        sdk_config = config_class(
            provider=str(self.config.get("provider", "deepseek-official")),
            model=str(self.config.get("model", "policy")),
            max_tokens=int(self.config["max_new_tokens"]),
            cwd=str(workspace_dir),
            runtime_cwd=str(artifact_dir if workspace is not None else workspace_dir),
            session_root=str(session_root),
            env=environment,
            runtime_bin=None if runtime_bin is None else str(runtime_bin),
            launch_args_override=launch_args,
            request_timeout_seconds=float(self.config.get("timeout_seconds", 1800.0)),
            shutdown_timeout_seconds=float(
                self.config.get("shutdown_timeout_seconds", 5.0)
            ),
            base_url=model_url or self.gateway_url,
            api_key=session_id,
        )
        prompt_text = self.prompt.messages[-1].content
        instruction = str(self.config.get("instruction_template", "{prompt}")).format(
            prompt=prompt_text
        )
        if workspace is not None:
            instruction = _REPOSITORY_TOOL_INSTRUCTION + instruction
        with harness_class(sdk_config) as harness:
            result = harness.run(instruction, session_id=session_id)
        events = [_json_value(event) for event in result.events]
        with (artifact_dir / "deepseek-events.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for event in events:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        summary = {
            "session_id": str(result.session_id),
            "final_response": str(result.final_response),
            "finish_reason": result.finish_reason,
            "notifications": [_json_value(item) for item in result.notifications],
        }
        (artifact_dir / "deepseek-result.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        final_answer = (
            "" if result.final_response is None else str(result.final_response)
        )
        if not final_answer and result.finish_reason not in {"error", "aborted"}:
            raise RuntimeError(
                f"DeepSeek Harness did not return a final response; see {artifact_dir}"
            )
        return final_answer, result.finish_reason, events


class DeepSeekProgramFactory(HarnessProgramFactory):
    """Create policy-identity-bound DeepSeek programs for ProgramAgentRunner."""

    program_type = DeepSeekAgentProgram
    label = "DeepSeek"
    include_admin_url = True


def _captured_tool_calls(captured):
    """Count captured tool calls while preserving malformed-record handling."""
    completions = captured.get("completions", [])
    actual_tool_calls = 0
    tool_names: list[str] = []
    if isinstance(completions, list):
        for completion in completions:
            if not isinstance(completion, Mapping):
                continue
            response = completion.get("response")
            choices = (
                response.get("choices")
                if isinstance(response, Mapping)
                else None
            )
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            message = (
                choice.get("message") if isinstance(choice, Mapping) else None
            )
            tool_calls = (
                message.get("tool_calls")
                if isinstance(message, Mapping)
                else None
            )
            if isinstance(tool_calls, list):
                actual_tool_calls += len(tool_calls)
                for tool_call in tool_calls:
                    function = (
                        tool_call.get("function")
                        if isinstance(tool_call, Mapping)
                        else None
                    )
                    name = (
                        function.get("name")
                        if isinstance(function, Mapping)
                        else None
                    )
                    tool_names.append(str(name or ""))
    return actual_tool_calls, tool_names
