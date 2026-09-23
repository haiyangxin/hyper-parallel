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
"""Management isolation and bounded transport contracts for external agent gateways."""

from io import BytesIO
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import urllib.error
import urllib.request

import pytest

from rl.agentic.codex import gateway as codex
from rl.agentic.ds_harness import gateway as deepseek
from tests.common.mark_utils import arg_mark


_MARK = arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
                 card_mark="onecard", essential_mark="essential")
_GATEWAYS = [(codex.CodexGateway, "/v1/responses"),
             (deepseek.DeepSeekGateway, "/v1/chat/completions")]


def _http(gateway: Any, method: str, path: str, token: str | None = None,
          body: bytes | None = None) -> tuple[int, dict]:
    """Use real local HTTP, retaining error status and decoded response bodies."""
    headers = {} if token is None else {"Authorization": "Bearer " + token}
    request = urllib.request.Request(
        f"http://{gateway.address[0]}:{gateway.address[1]}{path}",
        method=method, data=body, headers=headers,
    )
    try:
        response = urllib.request.urlopen(request, timeout=3)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.loads(response.read())


@_MARK
@pytest.mark.parametrize("gateway_type", [codex.CodexGateway, deepseek.DeepSeekGateway])
def test_management_auth_precedes_body_and_cleanup_blocks_late_registration(gateway_type: Any) -> None:
    """A candidate cannot inspect, release, register, or fail sessions using its model token."""
    gateway = gateway_type("127.0.0.1", 0, "http://unused", "test", 2)
    gateway.start()
    try:
        for method, path in [("GET", "/internal/sessions/s"), ("DELETE", "/internal/sessions/s"),
                             ("POST", "/internal/sessions"), ("POST", "/internal/sessions/s/failure")]:
            for token in (None, "s"):
                status, body = _http(gateway, method, path, token, b"invalid JSON")
                assert status == 401
                assert gateway.admin_token not in json.dumps(body)
        payload = json.dumps({"session_id": "s", "policy_version": 2, "max_completions": 2}).encode()
        assert _http(gateway, "POST", "/internal/sessions", gateway.admin_token, payload)[0] == 201
        assert _http(gateway, "GET", "/internal/sessions/s", "s")[0] == 401
        assert _http(gateway, "POST", "/internal/sessions/s/failure", gateway.admin_token, b"{}")[0] == 200
        status, snapshot = _http(gateway, "GET", "/internal/sessions/s", gateway.admin_token)
        assert status == 200 and snapshot["failure"]["trainable"] is False
        for session_id in ("s", "pending"):
            for _ in range(2):
                assert _http(gateway, "DELETE", "/internal/sessions/" + session_id, gateway.admin_token)[0] == 200
            late = json.dumps({"session_id": session_id, "policy_version": 2, "max_completions": 2}).encode()
            assert _http(gateway, "POST", "/internal/sessions", gateway.admin_token, late)[0] == 400
    finally:
        gateway.close()


@_MARK
@pytest.mark.parametrize("gateway_type,route", _GATEWAYS)
@pytest.mark.parametrize("body", [b"{invalid", b"x" * 257])
def test_bad_model_body_invalidates_known_session(gateway_type: Any, route: str, body: bytes) -> None:
    """Rejected bodies are infrastructure failures, never silently trainable empty traces."""
    gateway = gateway_type("127.0.0.1", 0, "http://unused", "test", 2, max_request_bytes=256)
    gateway.start()
    try:
        payload = json.dumps({"session_id": "s", "policy_version": 1, "max_completions": 1}).encode()
        assert _http(gateway, "POST", "/internal/sessions", gateway.admin_token, payload)[0] == 201
        assert _http(gateway, "POST", route, "s", body)[0] == 502
        _, snapshot = _http(gateway, "GET", "/internal/sessions/s", gateway.admin_token)
        assert snapshot["completions"] == []
        assert snapshot["failure"]["failure_origin"] == "infrastructure"
        assert snapshot["failure"]["trainable"] is False
    finally:
        gateway.close()


def _handler(module: Any) -> Any:
    """Construct the real request processing path with a controlled client transport."""
    handler = object.__new__(module._Handler)
    state = module._State("http://unused", "test", 2)
    state.register("s", {"policy_version": 3, "max_completions": 2})
    handler.server = SimpleNamespace(state=state, max_request_bytes=1024, max_response_bytes=4096)
    handler.headers = {"Authorization": "Bearer s"}
    handler.send_response = lambda *_args: None
    handler.send_header = lambda *_args: None
    handler.end_headers = lambda: None
    return handler


def _repository_request(prompt: str = "question") -> dict[str, Any]:
    """Model the two context blocks emitted by the pinned candidate Codex CLI."""
    return {
        "instructions": "Repair the repository and run a focused test.",
        "client_metadata": {"x-codex-turn-metadata": json.dumps({"request_kind": "turn"})},
        "tools": [{"type": "function", "name": "exec_command", "description": "Run a shell command.",
                   "parameters": {"type": "object", "properties": {
                       "cmd": {"type": "string"}, "sandbox_permissions": {"type": "string"},
                       "justification": {"type": "string"}, "prefix_rule": {"type": "array"}},
                       "required": ["cmd"], "additionalProperties": False}},
                  {"type": "function", "name": "write_stdin", "description": "Write to a running session.",
                   "parameters": {"type": "object", "properties": {"session_id": {"type": "number"}},
                                  "required": ["session_id"], "additionalProperties": False}}],
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": (
                "<permissions instructions>\n"
                "Filesystem sandboxing defines which files can be read or written. `sandbox_mode` is "
                "`danger-full-access`: No filesystem sandboxing - all commands are permitted. "
                "Network access is enabled.\n"
                "Approval policy is currently never. Do not provide the `sandbox_permissions` for any reason, "
                "commands will be rejected.\n"
                "</permissions instructions>"
            )}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": (
                "<environment_context>\n"
                "  <cwd>/workspace</cwd>\n"
                "  <shell>bash</shell>\n"
                "  <permission_profile type=\"disabled\"><file_system type=\"unrestricted\" />"
                "</permission_profile>\n"
                "</environment_context>"
            )}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
    }


def _sampled_repository_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """Tie one parsed tool action to immutable sampled token and parser evidence."""
    raw = '<tool_call>{"name":"exec_command","arguments":' + json.dumps(arguments) + '}</tool_call>'
    parsed = [{"id": "call-1", "type": "function", "function": {
        "name": "exec_command", "arguments": json.dumps(arguments),
    }}]
    return {"id": "sampled-tool", "model": "test", "prompt_token_ids": [9, 10],
            "choices": [{"index": 0, "token_ids": [11, 12], "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": None, "tool_calls": parsed},
                         "logprobs": {"content": [{"token_id": 11, "logprob": -0.2},
                                                   {"token_id": 12, "logprob": -0.3}]}}],
            "hyper_tool_protocol": [{"parser_input": raw, "engine_text": raw, "decoded_tokens": raw,
                                     "token_ids": [11, 12], "parser_result": {"tool_calls": parsed}}]}


@_MARK
def test_repository_rejects_undeclared_stdin_and_resamples_with_cmd_heredoc_feedback() -> None:
    """A parser-verified rejected action receives cmd heredoc feedback before resampling."""
    handler = _handler(codex)
    handler.server.state.get("s").repository_task = True
    handler.wfile = BytesIO()
    bad = _sampled_repository_tool({"cmd": "python /opt/hyper-codex-home/checked_patch.py",
                                    "stdin": "*** Begin Patch\nsecret patch\n*** End Patch"})
    good = {"id": "final", "model": "test", "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "Need a cmd heredoc."}, "finish_reason": "stop"}]}
    with patch.object(handler, "_backend_request", side_effect=[bad, good]) as backend:
        handler._proxy_responses(_repository_request())
    records = handler.server.state.snapshot("s")["completions"]
    assert backend.call_count == 2
    assert records[0]["response"] == bad
    assert records[0]["response"]["choices"][0]["token_ids"] == [11, 12]
    assert records[0]["metadata"]["failure_origin"] == "model"
    assert records[0]["metadata"]["failure_reason"] == "undeclared_tool_argument"
    assert records[0]["metadata"]["trainable"] is True
    assert records[0]["metadata"]["invalid_arguments"] == ["stdin"]
    assert records[1]["request"]["messages"][-2]["content"] == bad["hyper_tool_protocol"][0]["engine_text"]
    feedback = records[1]["request"]["messages"][-1]["content"]
    assert "stdin" in feedback and "cmd shell command" in feedback
    assert "<<'PATCH'\n*** Begin Patch" in feedback
    assert "secret patch" not in feedback
    assert records[1]["metadata"]["failure_origin"] is None


@_MARK
@pytest.mark.parametrize("arguments,reason", [({}, "missing_tool_argument"),
                                                  ({"cmd": False}, "invalid_tool_argument_type")])
def test_repository_validates_required_and_typed_tool_arguments(arguments: dict, reason: str) -> None:
    """The forwarded closed schema also rejects missing or mistyped declared fields."""
    handler = _handler(codex)
    session = handler.server.state.get("s")
    session.repository_task = True
    session.max_completions = 1
    bad = _sampled_repository_tool(arguments)
    with patch.object(handler, "_backend_request", return_value=bad), pytest.raises(codex.ToolProtocolFailure):
        handler._proxy_responses(_repository_request())
    snapshot = handler.server.state.snapshot("s")
    assert snapshot["completions"][0]["response"] == bad
    assert snapshot["completions"][0]["metadata"]["failure_reason"] == reason
    assert snapshot["failure"]["failure_reason"] == "tool_format_budget_exhausted"
    assert snapshot["failure"]["trainable"] is True


@_MARK
@pytest.mark.parametrize("repository", [False, True])
def test_repository_schema_validation_preserves_text_and_untrusted_evidence(repository: bool) -> None:
    """Text stays unchanged; missing token identity cannot become a model format error."""
    handler = _handler(codex)
    handler.server.state.get("s").repository_task = repository
    handler.wfile = BytesIO()
    sampled = _sampled_repository_tool({"cmd": "true", "stdin": "ignored"})
    if repository:
        sampled["hyper_tool_protocol"][0]["token_ids"] = [99]
    with patch.object(handler, "_backend_request", return_value=sampled) as backend:
        if repository:
            with pytest.raises(codex.ToolProtocolFailure):
                handler._proxy_responses(_repository_request())
        else:
            handler._proxy_responses(_repository_request())
    assert backend.call_count == 1
    snapshot = handler.server.state.snapshot("s")
    metadata = snapshot["completions"][0]["metadata"]
    if repository:
        assert metadata["failure_origin"] == "unknown" and metadata["trainable"] is False
    else:
        assert metadata["failure_origin"] is None and metadata["trainable"] is True


@_MARK
@pytest.mark.parametrize("repository", [False, True])
def test_repository_context_corrected_only_for_model_and_preserves_original(repository: bool) -> None:
    """The real request keeps evidence while the backend sees only truthful Docker permissions."""
    handler = _handler(codex)
    handler.server.state.get("s").repository_task = repository
    handler.wfile = BytesIO()
    request = _repository_request()
    original = json.loads(json.dumps(request))
    raw = {"id": "c", "model": "test", "choices": [{"index": 0,
           "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]}
    with patch.object(handler, "_backend_request", return_value=raw) as backend:
        handler._proxy_responses(request)
    record = handler.server.state.snapshot("s")["completions"][0]
    assert request == original == record["original_request"]
    assert backend.call_args.args[0] == record["request"]
    system = record["request"]["messages"][0]["content"]
    environment = record["request"]["messages"][1]["content"]
    if repository:
        assert "network=none" in system and "No Docker socket or NPU" in system
        assert "a separate trusted grader" in system
        assert "Network access is enabled" not in system
        assert 'type="candidate-container"' in environment
        assert 'type="unrestricted"' not in environment
        original_exec = record["original_request"]["tools"][0]["parameters"]["properties"]
        visible_exec = record["request"]["tools"][0]["function"]
        assert "sandbox_permissions" in original_exec and "justification" in original_exec
        assert "sandbox_permissions" not in visible_exec["parameters"]["properties"]
        assert "justification" not in visible_exec["parameters"]["properties"]
        assert "prefix_rule" not in visible_exec["parameters"]["properties"]
        assert "checked_patch.py" in visible_exec["description"]
    else:
        assert "Network access is enabled" in system
        assert 'type="unrestricted"' in environment
        assert "sandbox_permissions" in record["request"]["tools"][0]["function"]["parameters"]["properties"]
    assert record["request"]["messages"][-1]["content"] == "question"


@_MARK
@pytest.mark.parametrize("invalid", ["missing", "permissions", "environment"])
def test_repository_context_rejects_unrecognized_cli_blocks(invalid: str) -> None:
    """An unrecognized CLI injection is an infrastructure failure before sampling."""
    handler = _handler(codex)
    handler.server.state.get("s").repository_task = True
    request = _repository_request()
    if invalid == "missing":
        request["input"].pop(0)
    elif invalid == "permissions":
        request["input"][0]["content"][0]["text"] += " unknown"
    else:
        request["input"][1]["content"][0]["text"] = request["input"][1]["content"][0]["text"].replace(
            'type="unrestricted"', 'type="unknown"'
        )
    with patch.object(handler, "_backend_request") as backend, pytest.raises(ValueError, match="Codex"):
        handler._proxy_responses(request)
    backend.assert_not_called()
    snapshot = handler.server.state.snapshot("s")
    assert snapshot["completions"] == []
    assert snapshot["failure"]["failure_origin"] == "infrastructure"
    assert snapshot["failure"]["trainable"] is False


@_MARK
def test_repository_context_rejects_extra_cli_tool() -> None:
    """CLI tool drift is an infrastructure error before the backend sees a request."""
    handler = _handler(codex)
    handler.server.state.get("s").repository_task = True
    request = _repository_request()
    request["tools"].append({"type": "function", "name": "extra_tool", "description": "extra",
                             "parameters": {"type": "object", "properties": {}}})
    with patch.object(handler, "_backend_request") as backend, pytest.raises(ValueError, match="tool set"):
        handler._proxy_responses(request)
    backend.assert_not_called()
    snapshot = handler.server.state.snapshot("s")
    assert snapshot["completions"] == []
    assert snapshot["failure"]["failure_origin"] == "infrastructure"
    assert snapshot["failure"]["trainable"] is False


@_MARK
def test_repository_compaction_has_no_tools_and_preserves_sampled_request() -> None:
    """A pinned CLI summary request may run without tools while retaining truthful context."""
    handler = _handler(codex)
    handler.server.state.get("s").repository_task = True
    handler.wfile = BytesIO()
    request = _repository_request()
    request["client_metadata"]["x-codex-turn-metadata"] = json.dumps({"request_kind": "compaction"})
    request["tools"] = []
    request["input"][-1]["content"] = [{"type": "input_text", "text": codex.REPOSITORY_COMPACT_PROMPT}]
    original = json.loads(json.dumps(request))
    raw = {"id": "summary", "model": "test", "choices": [{"index": 0,
           "message": {"role": "assistant", "content": "State summary"}, "finish_reason": "stop"}]}
    with patch.object(handler, "_backend_request", return_value=raw) as backend:
        handler._proxy_responses(request)
    record = handler.server.state.snapshot("s")["completions"][0]
    assert request == original == record["original_request"]
    assert backend.call_args.args[0] == record["request"]
    assert record["request"].get("tools") in (None, [])
    assert record["request"]["max_tokens"] == 1024
    assert "network=none" in record["request"]["messages"][0]["content"]
    assert 'type="candidate-container"' in record["request"]["messages"][1]["content"]


@_MARK
@pytest.mark.parametrize("finish_reason,content", [("length", "incomplete"), ("stop", "")])
def test_repository_compaction_rejects_incomplete_summary(finish_reason: str, content: str) -> None:
    """Never replace CLI history with a truncated or empty summary."""
    handler = _handler(codex)
    handler.server.state.get("s").repository_task = True
    request = _repository_request()
    request["client_metadata"]["x-codex-turn-metadata"] = json.dumps({"request_kind": "compaction"})
    request["tools"] = []
    request["input"][-1]["content"] = [{"type": "input_text", "text": codex.REPOSITORY_COMPACT_PROMPT}]
    raw = {"id": "summary", "model": "test", "prompt_token_ids": [10],
           "choices": [{"index": 0, "token_ids": [11], "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason}]}
    with patch.object(handler, "_backend_request", return_value=raw), pytest.raises(codex.ToolProtocolFailure):
        handler._proxy_responses(request)
    snapshot = handler.server.state.snapshot("s")
    assert len(snapshot["completions"]) == 1
    assert snapshot["completions"][0]["response"] == raw
    assert snapshot["failure"]["failure_origin"] == "infrastructure"
    assert snapshot["failure"]["trainable"] is False


@_MARK
@pytest.mark.parametrize("invalid", ["kind", "tools", "prompt"])
def test_repository_compaction_rejects_contract_drift(invalid: str) -> None:
    """A no-tool request is accepted only as the pinned CLI's summary operation."""
    handler = _handler(codex)
    handler.server.state.get("s").repository_task = True
    request = _repository_request()
    request["client_metadata"]["x-codex-turn-metadata"] = json.dumps({"request_kind": "compaction"})
    request["tools"] = []
    request["input"][-1]["content"] = [{"type": "input_text", "text": codex.REPOSITORY_COMPACT_PROMPT}]
    if invalid == "kind":
        request["client_metadata"]["x-codex-turn-metadata"] = json.dumps({"request_kind": "turn"})
    elif invalid == "tools":
        request["tools"] = _repository_request()["tools"]
    else:
        request["input"][-1]["content"][0]["text"] += " changed"
    with patch.object(handler, "_backend_request") as backend, pytest.raises(ValueError, match="Codex"):
        handler._proxy_responses(request)
    backend.assert_not_called()
    snapshot = handler.server.state.snapshot("s")
    assert snapshot["completions"] == []
    assert snapshot["failure"]["failure_origin"] == "infrastructure"


@_MARK
@pytest.mark.parametrize("module", [codex, deepseek])
@pytest.mark.parametrize("failure", [BrokenPipeError, ConnectionResetError])
def test_sse_disconnect_retains_completion_and_rejects_episode(module: Any, failure: type) -> None:
    """A recorded sample cannot be trained after failure to deliver its SSE stream."""
    handler = _handler(module)
    raw = {"id": "c", "model": "test", "choices": [{"index": 0,
           "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]}
    stream = SimpleNamespace(write=lambda _data: (_ for _ in ()).throw(failure("disconnect")), flush=lambda: None)
    handler.wfile = stream
    handler._backend_request = lambda _payload: raw
    if module is codex:
        body = {"input": "question", "stream": True}
        process = handler._proxy_responses
    else:
        body = {"messages": [{"role": "user", "content": "question"}], "stream": True}
        process = handler._proxy_chat
    with pytest.raises(failure):
        process(body)
    snapshot = handler.server.state.snapshot("s")
    assert snapshot["completions"][0]["response"] == raw
    assert snapshot["failure"]["failure_origin"] == "infrastructure"
    assert snapshot["failure"]["trainable"] is False


@_MARK
@pytest.mark.parametrize("module", [codex, deepseek])
def test_backend_limits_cover_success_and_http_error(module: Any) -> None:
    """Neither ordinary nor error responses can bypass the backend byte limit."""
    handler = _handler(module)
    handler.server.max_response_bytes = 16
    for response in (BytesIO(b"x" * 17), urllib.error.HTTPError(
            "http://unused", 500, "error", {}, BytesIO(b"x" * 17))):
        if isinstance(response, urllib.error.HTTPError):
            mocked = patch.object(module.urllib.request, "urlopen", side_effect=response)
        else:
            mocked = patch.object(module.urllib.request, "urlopen", return_value=response)
        with mocked, pytest.raises(ValueError, match="max_response_bytes"):
            handler._backend_request({})
    assert handler._read_backend(BytesIO(b"x" * 16)) == b"x" * 16


@_MARK
@pytest.mark.parametrize("gateway_type", [codex.CodexGateway, deepseek.DeepSeekGateway])
def test_transport_settings_validate_before_binding(gateway_type: Any) -> None:
    """Bad byte limits and empty explicit credentials fail at construction."""
    for settings in ({"admin_token": ""}, {"max_request_bytes": 0}, {"max_response_bytes": True}):
        with pytest.raises(ValueError):
            gateway_type("127.0.0.1", 0, "http://unused", "test", 2, **settings)


@_MARK
@pytest.mark.parametrize("module", [codex, deepseek])
def test_response_limit_preserves_evidence_and_fails_delivery(module: Any) -> None:
    """The byte cap rejects an oversized client response without removing its raw completion."""
    handler = _handler(module)
    handler.server.max_response_bytes = 16
    handler.wfile = BytesIO()
    raw = {"id": "c", "model": "test", "choices": [{"index": 0,
           "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]}
    handler._backend_request = lambda _payload: raw
    for streaming in (False, True):
        state = module._State("http://unused", "test", 2)
        state.register("s", {"policy_version": 3, "max_completions": 2})
        handler.server.state = state
        body = {"stream": streaming}
        if module is codex:
            body["input"] = "question"
            process = handler._proxy_responses
        else:
            body["messages"] = [{"role": "user", "content": "question"}]
            process = handler._proxy_chat
        with pytest.raises(ValueError, match="max_response_bytes"):
            process(body)
        captured = state.snapshot("s")
        assert captured["completions"][0]["response"] == raw
        assert captured["failure"]["trainable"] is False
    assert handler.wfile.getvalue() == b""


@_MARK
@pytest.mark.parametrize("module", [codex, deepseek])
def test_slow_drip_read_obeys_total_deadline(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Frequent tiny chunks cannot renew either backend or client-body deadlines indefinitely."""
    handler = _handler(module)
    for length in (None, 100):
        ticks = iter([0.0, 0.0, 1.0, 1.0, 2.1])
        monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
        stream = SimpleNamespace(read1=lambda _size: b"x")
        with pytest.raises(TimeoutError, match="request_timeout"):
            handler._read_body(stream, 1024, length)


@_MARK
@pytest.mark.parametrize("repository", [False, True])
def test_completed_repository_budget_is_structured_and_text_budget_still_fails(repository: bool) -> None:
    """Only trusted repository registration turns fully captured call limits into a terminal state."""
    gateway = codex.CodexGateway("127.0.0.1", 0, "http://unused", "test", 2)
    raw = {"id": "c", "model": "test", "choices": [{"index": 0,
           "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]}
    gateway.start()
    try:
        registration = json.dumps({"session_id": "s", "policy_version": 3, "max_completions": 1,
                                   "repository_task": repository}).encode()
        assert _http(gateway, "POST", "/internal/sessions", gateway.admin_token, registration)[0] == 201
        request = json.dumps(_repository_request() if repository else {"input": "question"}).encode()
        with patch.object(codex._Handler, "_backend_request", return_value=raw) as backend:
            assert _http(gateway, "POST", "/v1/responses", "s", request)[0] == 200
            status, result = _http(gateway, "POST", "/v1/responses", "s", request)
            assert status == (409 if repository else 502)
            assert backend.call_count == 1
            if repository:
                assert result["error"]["type"] == "completion_budget_exhausted"
                assert result["termination"] == {"reason": "max_completions", "limit": 1,
                                                 "completed": 1, "policy_version": 3}
                assert _http(gateway, "POST", "/v1/responses", "s", request) == (status, result)
        _, snapshot = _http(gateway, "GET", "/internal/sessions/s", gateway.admin_token)
        assert len(snapshot["completions"]) == 1
        if repository:
            assert snapshot["failure"] is None
            assert snapshot["termination"] == result["termination"]
            assert _http(gateway, "POST", "/v1/responses", "s", request)[0] == 409
        else:
            assert "termination" not in snapshot
            assert snapshot["failure"]["trainable"] is False
    finally:
        gateway.close()


@_MARK
def test_repository_budget_requires_completed_reservations_and_real_boolean_opt_in() -> None:
    """Reserved work is not a completed sample; failure always takes precedence over a budget."""
    state = codex._State("http://unused", "test", 2)
    for value in (1, "true", None):
        with pytest.raises(ValueError, match="repository_task"):
            state.register("invalid", {"policy_version": 3, "max_completions": 1, "repository_task": value})
    state.register("s", {"policy_version": 3, "max_completions": 1, "repository_task": True})
    session = state.reserve_completion("s")
    with pytest.raises(ValueError, match="max_completions"):
        state.reserve_completion("s")
    assert session.termination is None
    state.save_completion("s", {"response": {}})
    state.release_completion(session)
    with pytest.raises(codex.CompletionBudgetExhausted):
        state.reserve_completion("s")
    state.record_failure(session, {"failure_origin": "infrastructure", "trainable": False})
    with pytest.raises(ValueError, match="failed"):
        state.reserve_completion("s")
    assert state.snapshot("s")["failure"]["trainable"] is False


@_MARK
def test_budget_response_delivery_failure_invalidates_the_terminal_session() -> None:
    """A typed budget cannot conceal a broken delivery to the candidate CLI."""
    handler = _handler(codex)
    state = handler.server.state
    session = state.get("s")
    session.repository_task = True
    session.completions = [{"ordinal": 0}, {"ordinal": 1}]
    handler.wfile = SimpleNamespace(write=lambda _data: (_ for _ in ()).throw(BrokenPipeError("disconnect")))
    with pytest.raises(BrokenPipeError):
        handler._proxy_responses(_repository_request())
    snapshot = state.snapshot("s")
    assert snapshot["termination"]["reason"] == "max_completions"
    assert snapshot["failure"]["failure_origin"] == "infrastructure"
    assert snapshot["failure"]["trainable"] is False
    assert session.inflight_requests == 0


@_MARK
@pytest.mark.parametrize("recovers", [False, True])
def test_repository_format_resampling_keeps_its_existing_terminal_attribution(recovers: bool) -> None:
    """Format-budget failure stays model zero; recovered calls may later reach normal task budget."""
    handler = _handler(codex)
    state = handler.server.state
    session = state.get("s")
    session.repository_task = True
    session.max_completions = 2 if recovers else 1
    handler.wfile = BytesIO()
    raw = {"id": "c", "model": "test", "choices": [{"index": 0,
           "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
           "hyper_tool_protocol": [{"engine_text": "malformed tool"}]}
    handler._backend_request = lambda _payload: raw
    malformed = {"failure_origin": "model", "failure_reason": "invalid_json", "trainable": True}
    outcomes = [malformed, {"failure_origin": None, "trainable": True}] if recovers else [malformed]
    with patch.object(codex, "inspect_tool_response", side_effect=outcomes):
        if recovers:
            handler._proxy_responses(_repository_request())
            handler._proxy_responses(_repository_request("continue"))
            assert session.termination["reason"] == "max_completions"
            assert session.failure is None
        else:
            with pytest.raises(codex.ToolProtocolFailure):
                handler._proxy_responses(_repository_request())
            assert session.termination is None
            assert session.failure["failure_reason"] == "tool_format_budget_exhausted"
    assert len(session.completions) == session.max_completions
