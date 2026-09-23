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
"""DeepSeek repository call budgets preserve a frozen, scoreable episode."""

from __future__ import annotations

from http import HTTPStatus
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rl.agentic.ds_harness import gateway


def _handler(repository_task: bool) -> tuple[Any, gateway._State, list[tuple[HTTPStatus, dict[str, Any]]]]:
    state = gateway._State("http://unused", "policy", 2)
    state.register("session", {
        "policy_version": 7,
        "max_completions": 1,
        "repository_task": repository_task,
    })
    handler = object.__new__(gateway._Handler)
    handler.server = SimpleNamespace(state=state, max_response_bytes=4096)
    handler.headers = {"Authorization": "Bearer session"}
    replies: list[tuple[HTTPStatus, dict[str, Any]]] = []
    handler._json = lambda status, payload: replies.append((status, payload))
    handler._backend_request = lambda _payload: {
        "choices": [{"message": {"role": "assistant", "content": "continue"}, "finish_reason": "stop"}],
    }
    return handler, state, replies


def _request() -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": "repair the repository"}], "stream": False}


def test_repository_budget_returns_structured_terminal_without_backend_retry() -> None:
    """The extra SDK request signals a completed budget and preserves the sampled call."""
    handler, state, replies = _handler(True)
    backend_calls = 0

    def backend(_payload: dict[str, Any]) -> dict[str, Any]:
        """Supply a controlled backend response for the gateway contract."""
        nonlocal backend_calls
        backend_calls += 1
        return {"choices": [{"message": {"role": "assistant", "content": "continue"}, "finish_reason": "stop"}]}

    handler._backend_request = backend
    handler._proxy_chat(_request())
    handler._proxy_chat(_request())
    handler._proxy_chat(_request())

    terminal = {"reason": "max_completions", "limit": 1, "completed": 1, "policy_version": 7}
    assert backend_calls == 1
    assert replies[0][0] == HTTPStatus.OK
    assert replies[1:] == [(HTTPStatus.CONFLICT, {
        "error": {"type": "completion_budget_exhausted",
                  "message": "DeepSeek repository reached max_completions=1"},
        "termination": terminal,
    })] * 2
    snapshot = state.snapshot("session")
    assert len(snapshot["completions"]) == 1
    assert snapshot["completions"][0]["metadata"]["policy_version"] == 7
    assert snapshot["failure"] is None
    assert snapshot["termination"] == terminal


def test_text_budget_keeps_existing_model_failure() -> None:
    """Text episodes retain their established trainable budget failure."""
    handler, state, replies = _handler(False)
    handler._proxy_chat(_request())
    with pytest.raises(ValueError, match="max_completions=1"):
        handler._proxy_chat(_request())
    snapshot = state.snapshot("session")
    assert [status for status, _ in replies] == [HTTPStatus.OK]
    assert "termination" not in snapshot
    assert snapshot["failure"] == {
        "failure_origin": "model", "failure_reason": "call_budget_exhausted", "trainable": True,
    }


def test_budget_response_disconnect_overrides_normal_terminal() -> None:
    """A failed delivery cannot turn a sampled but undelivered episode into reward evidence."""
    handler, state, _replies = _handler(True)
    handler._proxy_chat(_request())

    def disconnect(_status: HTTPStatus, _payload: dict[str, Any]) -> None:
        """Simulate candidate disconnection during terminal response delivery."""
        raise BrokenPipeError("candidate disconnected")

    handler._json = disconnect
    with pytest.raises(BrokenPipeError, match="candidate disconnected"):
        handler._proxy_chat(_request())
    snapshot = state.snapshot("session")
    assert snapshot["termination"]["completed"] == 1
    assert snapshot["failure"]["failure_origin"] == "infrastructure"
    assert snapshot["failure"]["trainable"] is False


def test_budget_response_size_failure_is_infrastructure() -> None:
    """An unparseable terminal response cannot authorize freezing the candidate."""
    handler, state, _replies = _handler(True)
    handler._proxy_chat(_request())
    handler.server.max_response_bytes = 1
    with pytest.raises(ValueError, match="budget response exceeds max_response_bytes"):
        handler._proxy_chat(_request())
    snapshot = state.snapshot("session")
    assert snapshot["termination"]["completed"] == 1
    assert snapshot["failure"]["failure_origin"] == "infrastructure"
    assert snapshot["failure"]["trainable"] is False


def test_repository_registration_requires_boolean_mode() -> None:
    """Only controller registration may opt in to repository budget semantics."""
    state = gateway._State("http://unused", "policy", 2)
    with pytest.raises(ValueError, match="repository_task must be a boolean"):
        state.register("session", {"policy_version": 7, "max_completions": 1,
                                   "repository_task": "true"})


@pytest.mark.parametrize("name,arguments,reason", [
    ("not_declared", {"command": "echo hi", "description": "say hi"}, "undeclared_tool_name"),
    ("bash", {"command": "echo hi", "description": "say hi", "stdin": "ignored"}, "invalid_tool_schema"),
    ("bash", {"command": "echo hi"}, "invalid_tool_schema"),
    ("bash", {"command": 2, "description": "say hi"}, "invalid_tool_schema"),
])
def test_repository_tool_calls_match_model_visible_schema(name: str, arguments: dict, reason: str) -> None:
    """Extra or unknown model arguments fail before the SDK can silently ignore them."""
    request = {"tools": [{"type": "function", "function": {
        "name": "bash", "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "description": {"type": "string"}},
            "required": ["command", "description"], "additionalProperties": False},
    }}]}
    response = {"choices": [{"message": {"tool_calls": [{"function": {
        "name": name, "arguments": json.dumps(arguments),
    }}]}}]}
    outcome = gateway._repository_tool_argument_failure(response, request)
    assert outcome is not None and outcome["failure_origin"] == "model"
    assert outcome["trainable"] is True and outcome["failure_reason"] == reason


def test_optional_tool_without_required_arguments_is_valid() -> None:
    """The stock DS job_list schema omits required, even though bash-only is preferred."""
    request = {"tools": [{"function": {"name": "job_list", "parameters": {
        "type": "object", "properties": {}, "additionalProperties": False,
    }}}]}
    response = {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "job_list", "arguments": "{}",
    }}]}}]}
    assert gateway._repository_tool_argument_failure(response, request) is None


def test_backend_context_rejection_retains_bounded_exact_request(tmp_path: Path) -> None:
    """A vLLM context failure retains the offending prompt, without inventing an action."""
    state = gateway._State("http://unused", "policy", 2)
    state.register("session", {"policy_version": 7, "max_completions": 2,
                               "repository_task": True, "artifact_dir": str(tmp_path)})
    handler = object.__new__(gateway._Handler)
    handler.server = SimpleNamespace(state=state, max_response_bytes=4096)
    handler.headers = {"Authorization": "Bearer session"}
    request = {"messages": [{"role": "user", "content": "X" * 20000}], "stream": False}

    def rejected(_payload: dict) -> dict:
        """Simulate a backend context-limit rejection."""
        raise RuntimeError("vLLM chat completion failed with HTTP 400: context limit")

    handler._backend_request = rejected
    with pytest.raises(RuntimeError, match="HTTP 400"):
        handler._proxy_chat(request)
    events = [json.loads(line) for line in (tmp_path / "gateway-events.jsonl").read_text().splitlines()]
    rejected_event = next(event for event in events if event["type"] == "completion.backend_rejected")
    assert rejected_event["payload"]["original_request"] == request
    assert rejected_event["payload"]["model_request"]["messages"] == request["messages"]
    assert rejected_event["payload"]["original_sha256"]
    assert state.snapshot("session")["failure"]["failure_origin"] == "infrastructure"
    assert state.snapshot("session")["completions"] == []


def test_repository_gateway_rejects_extra_bash_field_before_sdk_execution(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The model sees a closed schema and an invalid sampled call becomes model failure."""
    handler, state, _replies = _handler(True)
    request = _request()
    request["tools"] = [{"type": "function", "function": {
        "name": "bash", "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "description": {"type": "string"}},
            "required": ["command", "description"]},
    }}]
    seen = []

    def backend(payload: dict) -> dict:
        """Supply a controlled backend response for the gateway contract."""
        seen.append(payload)
        return {"choices": [{"message": {"role": "assistant", "content": None,
                                         "tool_calls": [{"function": {"name": "bash", "arguments": json.dumps({
                                             "command": "python checked_patch.py", "description": "edit",
                                             "stdin": "silently ignored patch",
                                         })}}]}, "finish_reason": "tool_calls"}]}

    handler._backend_request = backend
    monkeypatch.setattr(gateway, "inspect_tool_response", lambda _response: {"failure_origin": None})
    with pytest.raises(RuntimeError, match="invalid_tool_schema"):
        handler._proxy_chat(request)
    assert seen[0]["tools"][0]["function"]["parameters"]["additionalProperties"] is False
    assert "additionalProperties" not in request["tools"][0]["function"]["parameters"]
    snapshot = state.snapshot("session")
    assert len(snapshot["completions"]) == 1
    raw_parameters = snapshot["completions"][0]["original_request"]["tools"][0]["function"]["parameters"]
    assert "additionalProperties" not in raw_parameters
    assert snapshot["failure"]["failure_origin"] == "model"
    assert snapshot["failure"]["failure_reason"] == "invalid_tool_schema"
    assert snapshot["failure"]["trainable"] is True
