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
"""Codex Responses recorder with strict training and explicit external inference modes."""

from __future__ import annotations

import http.client
import json
import logging
import secrets
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import unquote, urlparse

from rl.agentic.codex.protocol import CodexResponsesProtocol
from rl.tool_protocol import inspect_tool_response


logger = logging.getLogger(__name__)

_CLI_PERMISSIONS = (
    "<permissions instructions>\n"
    "Filesystem sandboxing defines which files can be read or written. `sandbox_mode` is "
    "`danger-full-access`: No filesystem sandboxing - all commands are permitted. Network access is enabled.\n"
    "Approval policy is currently never. Do not provide the `sandbox_permissions` for any reason, "
    "commands will be rejected.\n"
    "</permissions instructions>"
)
_REPOSITORY_PERMISSIONS = (
    "<permissions instructions>\n"
    "This task runs in an isolated candidate Docker container with network=none. "
    "No Docker socket or NPU device is available. Put submitted source edits only in /workspace "
    "and obey the task's narrower file allowlist. Use /tmp only for temporary files. "
    "No permission escalation is available. The controller freezes the final artifact, and a separate "
    "trusted grader checks its file allowlist and evaluates the patch.\n"
    "</permissions instructions>"
)
_CLI_PERMISSION_PROFILE = '<permission_profile type="disabled"><file_system type="unrestricted" />' \
                          '</permission_profile>'
_REPOSITORY_PERMISSION_PROFILE = (
    '<permission_profile type="candidate-container"><file_system type="container-root">'
    '<source_edit_scope>/workspace</source_edit_scope><scratch>/tmp</scratch></file_system>'
    '<network>none</network><docker_socket>unavailable</docker_socket><npu>unavailable</npu>'
    '</permission_profile>'
)
REPOSITORY_COMPACT_PROMPT = (
    "HYPER_RL_CODE_AGENT_COMPACTION_V2\n"
    "Write only a compact handoff of at most 160 words in at most six bullets. Do not call tools. "
    "Keep the repository issue and edit constraints, exact relevant source paths and function names, "
    "current verified edits, latest test results, unresolved blockers, and the next concrete action. "
    "Distinguish failed commands and unchanged files from successful edits. "
    "Do not quote source code, patches, command output, or the chronological transcript. "
    "Summarize the current state, not every previous attempt. Finish within this word limit."
)


def _repository_request_kind(original: dict[str, Any]) -> str:
    """Use the pinned CLI's explicit request kind, never infer it from tools alone."""
    metadata = original.get("client_metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("x-codex-turn-metadata"), str):
        raise ValueError("Repository Codex request omitted pinned CLI metadata")
    try:
        kind = json.loads(metadata["x-codex-turn-metadata"])["request_kind"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Repository Codex request has invalid pinned CLI metadata") from error
    if kind not in {"turn", "compaction"}:
        raise ValueError("Repository Codex request has unsupported pinned CLI kind")
    return kind


def _repository_model_context(original: dict[str, Any], transformed: dict[str, Any]) -> str:
    """Correct the pinned CLI's misleading container context only in the model request."""
    request_kind = _repository_request_kind(original)
    input_items = original.get("input")
    if not isinstance(input_items, list) or len(input_items) < 2:
        raise ValueError("Repository Codex request omitted pinned CLI context items")
    developer, environment = input_items[:2]
    if (not isinstance(developer, dict) or developer.get("type") != "message"
            or developer.get("role") != "developer" or not isinstance(developer.get("content"), list)):
        raise ValueError("Repository Codex developer context differs from pinned CLI format")
    if (not isinstance(environment, dict) or environment.get("type") != "message"
            or environment.get("role") != "user" or not isinstance(environment.get("content"), list)):
        raise ValueError("Repository Codex environment context differs from pinned CLI format")
    permission_blocks = [block for block in developer["content"] if isinstance(block, dict)
                         and isinstance(block.get("text"), str)
                         and ("<permissions instructions>" in block["text"]
                              or "</permissions instructions>" in block["text"])]
    if (len(permission_blocks) != 1 or permission_blocks[0].get("type") != "input_text"
            or permission_blocks[0]["text"] != _CLI_PERMISSIONS):
        raise ValueError("Repository Codex permissions block differs from pinned CLI format")
    environment_blocks = [block for block in environment["content"]
                          if isinstance(block, dict) and block.get("type") == "input_text"]
    if len(environment_blocks) != 1 or not isinstance(environment_blocks[0].get("text"), str):
        raise ValueError("Repository Codex environment block differs from pinned CLI format")
    environment_text = environment_blocks[0]["text"]
    if (not environment_text.startswith("<environment_context>\n")
            or not environment_text.endswith("</environment_context>")
            or environment_text.count("<environment_context>") != 1
            or environment_text.count("</environment_context>") != 1
            or environment_text.count("<permission_profile") != 1
            or environment_text.count("</permission_profile>") != 1
            or environment_text.count("<cwd>/workspace</cwd>") != 1
            or environment_text.count(_CLI_PERMISSION_PROFILE) != 1):
        raise ValueError("Repository Codex environment block differs from pinned CLI format")

    messages = transformed["messages"]
    if (not messages or messages[0].get("role") != "system"
            or messages[0]["content"].count(_CLI_PERMISSIONS) != 1):
        raise ValueError("Repository Codex permissions could not be located after translation")
    messages[0]["content"] = messages[0]["content"].replace(_CLI_PERMISSIONS, _REPOSITORY_PERMISSIONS)
    environment_messages = [message for message in messages
                            if message.get("role") == "user" and message.get("content") == environment_text]
    if len(environment_messages) != 1:
        raise ValueError("Repository Codex environment could not be located after translation")
    environment_messages[0]["content"] = environment_text.replace(
        _CLI_PERMISSION_PROFILE, _REPOSITORY_PERMISSION_PROFILE
    )
    tools = transformed.get("tools")
    if request_kind == "compaction":
        last = input_items[-1]
        if (original.get("tools") != [] or tools not in (None, [])
                or not isinstance(last, dict) or last.get("type") != "message"
                or last.get("role") != "user" or last.get("content") != [
                    {"type": "input_text", "text": REPOSITORY_COMPACT_PROMPT}]):
            raise ValueError("Repository Codex compaction request differs from pinned CLI contract")
        return request_kind
    if (not isinstance(tools, list) or len(tools) != 2
            or {tool.get("function", {}).get("name") for tool in tools if isinstance(tool, dict)}
            != {"exec_command", "write_stdin"}):
        raise ValueError("Repository Codex tool set differs from the two executable CLI tools")
    for tool in tools:
        function = tool.get("function", {})
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})
        if function.get("name") == "exec_command":
            if not isinstance(properties, dict):
                raise ValueError("Repository Codex exec_command schema has no properties")
            hidden = {"sandbox_permissions", "justification", "prefix_rule"}
            function["parameters"] = {
                **parameters,
                "properties": {name: value for name, value in properties.items() if name not in hidden},
            }
            function["description"] += (
                " For repository edits, set cmd to a shell heredoc, for example: "
                "python /opt/hyper-codex-home/checked_patch.py <<'PATCH'\n"
                "*** Begin Patch\n...\n*** End Patch\nPATCH\n"
                "Do not pass a separate stdin argument. Read changed-file hashes and diff; "
                "a shell exit code of zero alone does not prove a file changed."
            )
        _validate_repository_tool_schema(function)
    return request_kind


def _validate_repository_tool_schema(function: dict[str, Any]) -> None:
    """Require the pinned CLI's simple closed argument objects before sampling."""
    parameters = function.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise ValueError("Repository Codex tool parameters must be an object schema")
    properties = parameters.get("properties")
    required = parameters.get("required")
    if (not isinstance(properties, dict) or parameters.get("additionalProperties") is not False
            or not isinstance(required, list) or not all(isinstance(name, str) for name in required)
            or not set(required).issubset(properties)):
        raise ValueError("Repository Codex tool schema differs from pinned closed argument contract")
    if any(not isinstance(value, dict) or value.get("type") not in {"string", "number", "integer", "boolean"}
           for value in properties.values()):
        raise ValueError("Repository Codex tool schema has unsupported argument types")


def _repository_tool_argument_failure(response: dict[str, Any], transformed: dict[str, Any]) -> dict | None:
    """Classify only parser-verified tool calls that the forwarded CLI schema cannot execute."""
    tools = {tool["function"]["name"]: tool["function"]["parameters"] for tool in transformed["tools"]}
    calls = response["choices"][0]["message"].get("tool_calls") or []
    for call in calls:
        function = call["function"]
        name = function["name"]
        if name not in tools:
            return {"failure_origin": "model", "failure_reason": "undeclared_tool_name", "trainable": True,
                    "tool_feedback": f"Tool {name!r} is not declared. Nothing was executed. "
                                     "Use exec_command or write_stdin with their declared arguments."}
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict):
            return {"failure_origin": "model", "failure_reason": "invalid_tool_arguments", "trainable": True,
                    "tool_feedback": f"Tool {name!r} requires a JSON argument object. Nothing was executed."}
        schema = tools[name]
        properties = schema["properties"]
        extra = sorted(set(arguments) - set(properties))
        if extra:
            detail = ", ".join(extra)
            feedback = f"Tool {name!r} does not accept argument(s) {detail}. Nothing was executed."
            if name == "exec_command" and "stdin" in extra:
                feedback += (
                    " Put the complete patch in the cmd shell command using a heredoc: "
                    "python /opt/hyper-codex-home/checked_patch.py <<'PATCH'\n"
                    "*** Begin Patch\n...\n*** End Patch\nPATCH\n"
                    "Do not pass a separate stdin key."
                )
            return {"failure_origin": "model", "failure_reason": "undeclared_tool_argument", "trainable": True,
                    "tool_feedback": feedback, "tool_name": name, "invalid_arguments": extra}
        missing = sorted(set(schema["required"]) - set(arguments))
        if missing:
            return {"failure_origin": "model", "failure_reason": "missing_tool_argument", "trainable": True,
                    "tool_feedback": f"Tool {name!r} requires argument(s) {', '.join(missing)}. Nothing was executed."}
        for key, value in arguments.items():
            expected = properties[key]["type"]
            if expected == "string":
                valid = isinstance(value, str)
            elif expected == "boolean":
                valid = isinstance(value, bool)
            elif expected == "integer":
                valid = isinstance(value, int) and not isinstance(value, bool)
            else:
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not valid:
                return {"failure_origin": "model", "failure_reason": "invalid_tool_argument_type",
                        "trainable": True, "tool_feedback": f"Tool {name!r} requires {key!r} to be {expected}. "
                                                          "Nothing was executed."}
    return None


class ToolProtocolFailure(RuntimeError):
    """Carry an evidence-based terminal outcome without inventing a zero reward."""

    def __init__(self, outcome: dict[str, Any]) -> None:
        """Retain the classified terminal outcome for the harness."""
        super().__init__(outcome["failure_reason"])
        self.outcome = outcome


class CompletionBudgetExhausted(RuntimeError):
    """Identify a fully recorded repository call budget without inventing a model failure."""

    def __init__(self, termination: dict[str, Any]) -> None:
        """Retain the controller-owned terminal evidence for the HTTP response."""
        super().__init__(f"Repository session exhausted its {termination['limit']} model calls")
        self.termination = dict(termination)


@dataclass
class _Session:
    policy_version: int | None
    artifact_dir: Optional[Path]
    max_completions: int | None
    generation: dict[str, Any]
    repository_task: bool = False
    completions: list[dict[str, Any]] = field(default_factory=list)
    reserved_completions: int = 0
    inflight_requests: int = 0
    closing: bool = False
    failure: Optional[dict[str, Any]] = None
    termination: Optional[dict[str, Any]] = None


def _redact_backend_value(value: Any, credential: str) -> Any:
    """Redact decoded JSON strings, including object keys, before any persistence."""
    if isinstance(value, str):
        return value.replace(credential, "[REDACTED]")
    if isinstance(value, list):
        return [_redact_backend_value(item, credential) for item in value]
    if isinstance(value, dict):
        return {key.replace(credential, "[REDACTED]"): _redact_backend_value(item, credential)
                for key, item in value.items()}
    return value


def _backend_error_detail(body: bytes, credential: str | None) -> str:
    """Keep bounded provider diagnostics while withholding undecodable authenticated errors."""
    if credential is None:
        detail = body.decode("utf-8", errors="replace")
    else:
        try:
            decoded = json.loads(body)
            detail = "[redacted JSON; not raw] " + json.dumps(
                _redact_backend_value(decoded, credential), ensure_ascii=False,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            return "[authenticated backend error body withheld: non-JSON or cannot reliably redact]"
    return detail[:4096] + (" [truncated]" if len(detail) > 4096 else "")


class _NoBackendRedirect(urllib.request.HTTPRedirectHandler):
    """Keep controller credentials on the explicitly configured backend origin."""

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int,
                         msg: str, headers: Any, newurl: str) -> None:
        """Reject redirects rather than forwarding authentication to another endpoint."""
        return None


class _State:
    """Own registered gateway sessions and captured completion events."""
    def __init__(self, backend_url: str, model_name: str, request_timeout: float,
                 max_inflight_requests: int = 1, *, inference_only: bool = False,
                 backend_key_file: Path | None = None) -> None:
        """Initialize shared admission limits and trace storage."""
        if (isinstance(max_inflight_requests, bool) or not isinstance(max_inflight_requests, int)
                or max_inflight_requests <= 0):
            raise ValueError("Codex gateway max_inflight_requests must be a positive integer")
        if not isinstance(inference_only, bool):
            raise ValueError("inference_only must be a boolean")
        self.inference_only = inference_only
        self.backend_key = (Path(backend_key_file).read_text(encoding="utf-8").strip()
                            if backend_key_file is not None else None)
        if backend_key_file is not None and (not self.backend_key or any(
                char.isspace() for char in self.backend_key)):
            raise ValueError("Backend credential must be a non-empty single token")
        parsed_backend = urlparse(backend_url)
        if (parsed_backend.scheme not in {"http", "https"} or not parsed_backend.hostname
                or parsed_backend.username or parsed_backend.password
                or parsed_backend.query or parsed_backend.fragment):
            raise ValueError("Backend URL must be an HTTP origin without credentials, query or fragment")
        self.backend_url = backend_url.rstrip("/")
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.protocol = CodexResponsesProtocol()
        self.lock = threading.RLock()
        self.idle = threading.Condition(self.lock)
        self.backend_slots = threading.BoundedSemaphore(max_inflight_requests)
        self.sessions: dict[str, _Session] = {}
        self.released_sessions: set[str] = set()
        self.closing = False

    def register(self, session_id: str, payload: dict[str, Any]) -> None:
        """Validate and register a new gateway session and its artifact directory."""
        if not session_id:
            raise ValueError("Codex session ID must be non-empty")
        artifact_value = payload.get("artifact_dir")
        artifact_dir = Path(artifact_value).resolve() if isinstance(artifact_value, str) else None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
        if payload.get("inference_only", False) is not self.inference_only:
            raise ValueError("Session mode differs from controller gateway mode")
        policy_version = payload.get("policy_version")
        if self.inference_only:
            if policy_version is not None:
                raise ValueError("Inference sessions must not claim a policy version")
        elif (not isinstance(policy_version, int) or isinstance(policy_version, bool) or policy_version < 0):
            raise ValueError("Training sessions require a non-negative policy version")
        max_completions = payload.get("max_completions")
        if not (self.inference_only and max_completions is None) and (
                not isinstance(max_completions, int) or isinstance(max_completions, bool) or max_completions <= 0):
            raise ValueError("Codex session requires positive max_completions")
        repository_task = payload.get("repository_task", False)
        if not isinstance(repository_task, bool):
            raise ValueError("Codex session repository_task must be a boolean")
        generation = payload.get("generation", {})
        if not isinstance(generation, Mapping):
            raise ValueError("Codex session generation settings must be a mapping")
        allowed_generation = {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "ignore_eos",
        }
        unknown_generation = set(generation) - allowed_generation
        if unknown_generation:
            raise ValueError(
                "Unknown Codex generation settings: "
                f"{sorted(unknown_generation)}"
            )
        session = _Session(
            policy_version,
            artifact_dir,
            max_completions,
            dict(generation),
            repository_task=repository_task,
        )
        with self.lock:
            if self.closing:
                raise ValueError("Codex gateway is closing")
            if session_id in self.released_sessions:
                raise ValueError("Session was already released")
            if session_id in self.sessions:
                raise ValueError(f"Codex session already exists: {session_id}")
            self.sessions[session_id] = session
        self.event(session_id, "session.registered", payload)

    def get(self, session_id: str) -> _Session:
        """Resolve an existing session without inventing new lifecycle state."""
        with self.lock:
            try:
                return self.sessions[session_id]
            except KeyError as error:
                raise ValueError(f"Unknown Codex session: {session_id}") from error

    def remove(self, session_id: str) -> None:
        """Release one completed in-memory trace while retaining its artifacts."""
        with self.idle:
            self.released_sessions.add(session_id)
            if session_id not in self.sessions:
                return
            session = self.get(session_id)
            session.closing = True
            if not self.idle.wait_for(lambda: session.inflight_requests == 0, timeout=self.request_timeout):
                raise RuntimeError(f"Timed out draining Codex session: {session_id}")
            self.event(session_id, "session.released", {})
            self.sessions.pop(session_id, None)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        """Seal admission and return complete terminal evidence after existing handlers settle."""
        with self.idle:
            session = self.get(session_id)
            session.closing = True
            if not self.idle.wait_for(lambda: session.inflight_requests == 0, timeout=self.request_timeout):
                raise RuntimeError(f"Timed out reading in-flight Codex session: {session_id}")
            result = {"policy_version": session.policy_version, "completions": list(session.completions),
                      "failure": session.failure, "inference_only": self.inference_only}
            if session.termination is not None:
                result["termination"] = dict(session.termination)
            return result

    def record_failure(self, session: _Session, outcome: dict[str, Any]) -> None:
        """Never let a concurrent model failure hide a prior untrainable request failure."""
        with self.lock:
            if session.failure is None or (session.failure.get("trainable") is True
                                           and outcome.get("trainable") is not True):
                session.failure = dict(outcome)

    def begin_request(self, session_id: str) -> _Session:
        """Keep a session alive through its backend work and client response."""
        with self.idle:
            session = self.get(session_id)
            if self.closing or session.failure is not None or (session.closing and session.termination is None):
                raise ValueError(f"Codex session is closing or failed: {session_id}")
            session.inflight_requests += 1
            return session

    def finish_request(self, session: _Session) -> None:
        """Wake a draining session after this HTTP request has fully settled."""
        with self.idle:
            session.inflight_requests -= 1
            self.idle.notify_all()

    def drain(self) -> None:
        """Reject admission and wait for all owned HTTP handlers before shutdown."""
        with self.idle:
            self.closing = True
            for session in self.sessions.values():
                session.closing = True
            if not self.idle.wait_for(lambda: all(session.inflight_requests == 0 for session in self.sessions.values()),
                                      timeout=self.request_timeout):
                raise RuntimeError("Timed out draining Codex gateway requests")

    def reserve_completion(self, session_id: str) -> _Session:
        """Atomically reserve one model call, including calls waiting for admission."""
        with self.idle:
            session = self.get(session_id)
            if session.failure is not None or (session.closing and session.inflight_requests == 0):
                raise ValueError(f"Codex session is closing or failed: {session_id}")
            if session.termination is not None:
                raise CompletionBudgetExhausted(session.termination)
            if (session.repository_task and len(session.completions) == session.max_completions
                    and session.reserved_completions == 0):
                session.termination = {"reason": "max_completions", "limit": session.max_completions,
                                       "completed": len(session.completions), "policy_version": session.policy_version}
                self.event(session_id, "session.budget_exhausted", session.termination)
                raise CompletionBudgetExhausted(session.termination)
            if (session.max_completions is not None
                    and len(session.completions) + session.reserved_completions >= session.max_completions):
                raise ValueError(f"Codex session exceeded max_completions={session.max_completions}")
            session.reserved_completions += 1
            return session

    def release_completion(self, session: _Session) -> None:
        """Release a reserved slot after the call was recorded or failed."""
        with self.idle:
            if session.reserved_completions <= 0:
                raise RuntimeError("Codex completion reservation underflow")
            session.reserved_completions -= 1
            self.idle.notify_all()

    def save_completion(self, session_id: str, record: dict[str, Any]) -> None:
        """Append one raw completion and assign its stable trace ordinal."""
        session = self.get(session_id)
        with self.lock:
            record["ordinal"] = len(session.completions)
            session.completions.append(record)
        self.event(session_id, "completion.recorded", record)

    def event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        """Append a diagnostic event when the session has artifact storage."""
        try:
            session = self.get(session_id)
        except ValueError:
            return
        if session.artifact_dir is None:
            return
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "session_id": session_id,
            "payload": payload,
        }
        with self.lock:
            event_path = session.artifact_dir / "gateway-events.jsonl"
            with event_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")


class _Handler(BaseHTTPRequestHandler):
    """Translate HTTP gateway requests into versioned backend completions."""
    server: "_GatewayServer"

    def setup(self) -> None:
        """Bound disconnected client I/O as well as backend calls during shutdown."""
        super().setup()
        self.connection.settimeout(self.server.state.request_timeout)

    def do_GET(self) -> None:  # pylint: disable=C0103
        """Serve health and captured-session inspection."""
        path = urlparse(self.path).path
        if path.startswith("/internal/") and not self._authorize_admin():
            return
        if path == "/healthz":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        prefix = "/internal/sessions/"
        if path.startswith(prefix):
            session_id = unquote(path[len(prefix):])
            try:
                snapshot = self.server.state.snapshot(session_id)
            except ValueError as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
                return
            except RuntimeError as error:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
                return
            try:
                self._json(HTTPStatus.OK, snapshot)
            except ValueError as error:
                self._error(HTTPStatus.BAD_GATEWAY, str(error))
            return
        self._error(HTTPStatus.NOT_FOUND, "Unknown gateway route")

    def do_POST(self) -> None:  # pylint: disable=C0103
        """Register a session or proxy one Responses request."""
        path = urlparse(self.path).path
        if path.startswith("/internal/") and not self._authorize_admin():
            return
        if path.startswith("/internal/sessions/") and path.endswith("/failure"):
            try:
                self._request_json()
                session_id = unquote(path[len("/internal/sessions/"):-len("/failure")])
                session = self.server.state.get(session_id)
                self.server.state.record_failure(session, {
                    "failure_origin": "infrastructure", "trainable": False,
                    "failure_reason": "Candidate relay transport failed",
                })
            except (ValueError, OSError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            self._json(HTTPStatus.OK, {"recorded": True})
            return
        if path == "/internal/sessions":
            try:
                body = self._request_json()
            except (ValueError, OSError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            session_id = body.pop("session_id", None)
            try:
                self.server.state.register(str(session_id or ""), body)
            except (KeyError, TypeError, ValueError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            self._json(HTTPStatus.CREATED, {"session_id": session_id})
            return
        if path.rstrip("/") not in {"/responses", "/v1/responses"}:
            self._error(HTTPStatus.NOT_FOUND, "Unknown gateway route")
            return
        try:
            self._proxy_responses()
        except ToolProtocolFailure as error:
            self._json(HTTPStatus.BAD_GATEWAY, {"error": {
                "type": "tool_protocol_error", "message": str(error), **error.outcome,
            }})
        except (RuntimeError, ValueError, OSError) as error:
            logger.exception("Codex gateway request failed")
            self._error(HTTPStatus.BAD_GATEWAY, str(error))

    def do_DELETE(self) -> None:  # pylint: disable=C0103
        """Release a captured session after its trajectory is materialized."""
        path = urlparse(self.path).path
        if path.startswith("/internal/") and not self._authorize_admin():
            return
        prefix = "/internal/sessions/"
        if not path.startswith(prefix):
            self._error(HTTPStatus.NOT_FOUND, "Unknown gateway route")
            return
        session_id = unquote(path[len(prefix):])
        try:
            self.server.state.remove(session_id)
        except ValueError as error:
            self._error(HTTPStatus.NOT_FOUND, str(error))
            return
        except RuntimeError as error:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
            return
        self._json(HTTPStatus.OK, {"released": session_id})

    def _proxy_responses(self, original: dict[str, Any] | None = None) -> None:
        """Translate a Responses request and capture its backend completion."""
        session_id = self._bearer_token()
        session = self.server.state.begin_request(session_id)
        try:
            if original is None:
                original = self._request_json()
            self._respond_to_session(session_id, session, original)
        except CompletionBudgetExhausted as error:
            try:
                self._json(HTTPStatus.CONFLICT, {
                    "error": {"type": "completion_budget_exhausted", "message": str(error)},
                    "termination": error.termination,
                })
            except (RuntimeError, ValueError, OSError) as delivery_error:
                self.server.state.record_failure(session, {
                    "failure_origin": "infrastructure", "failure_reason": str(delivery_error), "trainable": False,
                })
                raise
        except ToolProtocolFailure as error:
            self.server.state.record_failure(session, error.outcome)
            raise
        except (RuntimeError, ValueError, OSError) as error:
            self.server.state.record_failure(session, {
                "failure_origin": "infrastructure", "failure_reason": str(error), "trainable": False,
            })
            raise
        finally:
            self.server.state.finish_request(session)

    @staticmethod
    def _inference_outcome(response: dict, transformed: dict, repository_task: bool,
                           request_kind: str) -> dict:
        """Validate structured API output without claiming training parser evidence."""
        outcome = {"failure_origin": None, "trainable": False}
        try:
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("Inference response requires exactly one choice")
            choice = choices[0]
            message = choice["message"]
            if not isinstance(message, dict) or choice.get("finish_reason") not in {"stop", "tool_calls"}:
                raise ValueError("Inference completion is incomplete or has an unsupported finish reason")
            if message.get("role") != "assistant":
                raise ValueError("Inference response must have assistant role")
            calls = message.get("tool_calls")
            calls = [] if calls is None else calls
            if not isinstance(calls, list):
                raise ValueError("Inference tool_calls must be a list")
            call_ids = set()
            for call in calls:
                if call.get("type") != "function" or not isinstance(call.get("id"), str) or not call["id"]:
                    raise ValueError("Inference tool call requires a function and non-empty ID")
                if call["id"] in call_ids:
                    raise ValueError("Inference tool call IDs must be unique")
                call_ids.add(call["id"])
            if repository_task and request_kind == "turn":
                failure = _repository_tool_argument_failure(response, transformed)
                if failure is not None:
                    raise ValueError(failure["failure_reason"])
            if not calls and not isinstance(message.get("content"), str):
                raise ValueError("Inference response omitted text and tool calls")
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            outcome.update(failure_origin="infrastructure", failure_reason=f"inference_protocol_error: {error}")
        return outcome

    def _model_completion(self, session_id: str, original: dict, transformed: dict,
                          request_kind: str = "turn") -> tuple[dict, dict]:
        """Admit one call and atomically convert its reservation into captured evidence."""
        state = self.server.state
        session = state.reserve_completion(session_id)
        record = None
        try:
            if not state.backend_slots.acquire(timeout=state.request_timeout):
                raise RuntimeError("Timed out waiting for Codex backend admission")
            try:
                if state.inference_only:
                    state.event(session_id, "request.started", {
                        "original_request": original, "request": transformed,
                        "inference_only": True, "trainable": False, "policy_version": None,
                    })
                response = self._backend_request(transformed)
            finally:
                state.backend_slots.release()
            outcome = (self._inference_outcome(response, transformed, session.repository_task, request_kind)
                       if state.inference_only else inspect_tool_response(response))
            if (session.repository_task and request_kind == "turn" and outcome["trainable"]
                    and outcome["failure_origin"] is None):
                try:
                    tool_failure = _repository_tool_argument_failure(response, transformed)
                except (KeyError, TypeError, ValueError):
                    outcome = {"failure_origin": "infrastructure", "trainable": False,
                               "failure_reason": "Verified tool call shape differs from parser evidence"}
                else:
                    if tool_failure is not None:
                        outcome = tool_failure
            if request_kind == "compaction":
                choices = response.get("choices", [])
                choice = choices[0] if isinstance(choices, list) and choices else {}
                message = choice.get("message", {}) if isinstance(choice, dict) else {}
                summary = message.get("content") if isinstance(message, dict) else None
                if choice.get("finish_reason") != "stop" or not isinstance(summary, str) or not summary.strip():
                    outcome = {"failure_origin": "infrastructure", "trainable": False,
                               "failure_reason": "Codex compaction did not return a complete non-empty summary"}
            record = {"timestamp": time.time(), "original_request": original, "request": transformed,
                      "response": response, "metadata": {"policy_version": session.policy_version,
                                                          "session_id": session_id,
                                                          "inference_only": state.inference_only, **outcome}}
            return response, outcome
        except Exception as error:
            if state.inference_only and record is None:
                state.event(session_id, "request.failed", {
                    "original_request": original, "request": transformed,
                    "error_type": type(error).__name__, "inference_only": True,
                    "trainable": False, "policy_version": None,
                })
            raise
        finally:
            with state.lock:
                try:
                    if record is not None:
                        state.save_completion(session_id, record)
                finally:
                    state.release_completion(session)

    def _respond_to_session(self, session_id: str, session: _Session, original: dict) -> None:
        """Keep rejected model actions intact while allowing explicitly budgeted resampling."""
        transformed = self.server.state.protocol.transform_request(
            original,
            self.server.state.model_name,
            request_training_evidence=not self.server.state.inference_only,
        )
        request_kind = _repository_model_context(original, transformed) if session.repository_task else "turn"
        transformed.update(session.generation)
        if session.repository_task and request_kind == "compaction":
            transformed["max_tokens"] = min(1024, transformed.get("max_tokens", 1024))
        while True:
            response, outcome = self._model_completion(session_id, original, transformed, request_kind)
            if self.server.state.inference_only:
                if outcome["failure_origin"] is not None:
                    raise ToolProtocolFailure(dict(outcome, model_calls=len(session.completions)))
                break
            if not outcome["trainable"]:
                raise ToolProtocolFailure(dict(outcome, model_calls=len(session.completions)))
            if outcome["failure_origin"] != "model":
                break
            if len(session.completions) >= session.max_completions:
                raise ToolProtocolFailure(dict(outcome, failure_reason="tool_format_budget_exhausted",
                                               model_calls=len(session.completions)))
            feedback = outcome.get("tool_feedback")
            transformed = {**transformed, "messages": [*transformed["messages"],
                {"role": "assistant", "content": response["hyper_tool_protocol"][0]["engine_text"]},
                {"role": "user", "content": feedback or "Tool format error: invalid JSON/schema. Nothing was executed. "
                 "Generate a new valid tool call or answer."},
            ]}
        result = self.server.state.protocol.transform_response(response, original)
        if bool(original.get("stream")):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Model-Calls", str(len(session.completions)))
            self.end_headers()
            self._write_sse_events(result)
            return
        self._json(HTTPStatus.OK, result, model_calls=len(session.completions))

    def _write_sse_events(self, result: dict[str, Any]) -> None:
        """Write one bounded Responses stream; delivery errors invalidate the episode."""
        deadline = time.monotonic() + self.server.state.request_timeout
        total = 0
        for event in self.server.state.protocol.stream_events(result):
            if time.monotonic() >= deadline:
                raise TimeoutError("Gateway SSE write exceeded request_timeout")
            payload = json.dumps(event, ensure_ascii=False).encode("utf-8")
            frame = b"event: " + str(event["type"]).encode("utf-8") + b"\n" + b"data: " + payload + b"\n\n"
            total += len(frame)
            if total > self.server.max_response_bytes:
                raise ValueError("Gateway SSE response exceeds max_response_bytes")
            self.wfile.write(frame)
            self.wfile.flush()

    def _backend_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON chat request to the configured inference backend."""
        data = json.dumps(payload).encode("utf-8")
        if len(data) > self.server.max_request_bytes:
            raise ValueError("Backend request exceeds max_request_bytes")
        headers = {"Content-Type": "application/json"}
        credential = self.server.state.backend_key
        if credential is not None:
            headers["Authorization"] = "Bearer " + credential
        request = urllib.request.Request(
            f"{self.server.state.backend_url}/v1/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            timeout = self.server.state.request_timeout
            open_request = (urllib.request.build_opener(_NoBackendRedirect()).open
                            if credential is not None else urllib.request.urlopen)
            with open_request(request, timeout=timeout) as response:
                body = self._read_backend(response)
        except urllib.error.HTTPError as error:
            detail = _backend_error_detail(self._read_backend(error), credential)
            raise RuntimeError(
                f"vLLM chat completion failed with HTTP {error.code}: {detail}"
            ) from (None if credential is not None else error)
        except urllib.error.URLError as error:
            raise RuntimeError(f"vLLM chat completion request failed: {error.reason}") from error
        except http.client.RemoteDisconnected as error:
            raise RuntimeError(
                "vLLM chat completion connection closed before a response"
            ) from error
        if credential is not None and credential.encode() in body:
            raise RuntimeError("Backend response contained a credential; response withheld")
        try:
            decoded = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("vLLM chat completion returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise RuntimeError("vLLM chat completion returned a non-object response")
        if credential is not None and credential in json.dumps(decoded, ensure_ascii=False):
            raise RuntimeError("Backend response contained a credential; response withheld")
        return decoded

    def _authorize_admin(self) -> bool:
        """Authenticate management requests before reading bodies or touching session state."""
        supplied = self.headers.get("Authorization", "")
        if secrets.compare_digest(supplied.encode(), ("Bearer " + self.server.admin_token).encode()):
            return True
        self._error(HTTPStatus.UNAUTHORIZED, "Management authentication required")
        return False

    def _read_body(self, stream: Any, limit: int, length: int | None = None) -> bytes:
        """Bound total read time as well as bytes, including clients that continually drip data."""
        deadline = time.monotonic() + self.server.state.request_timeout
        chunks = []
        total = 0
        while length is None or total < length:
            if time.monotonic() >= deadline:
                raise TimeoutError("Gateway body read exceeded request_timeout")
            size = min(65536, limit + 1 - total, length - total if length is not None else limit + 1)
            chunk = stream.read1(size)
            if time.monotonic() >= deadline:
                raise TimeoutError("Gateway body read exceeded request_timeout")
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ValueError("Gateway body exceeds max_response_bytes")
            chunks.append(chunk)
        if length is not None and total != length:
            raise ValueError("Request body ended before Content-Length")
        return b"".join(chunks)

    def _read_backend(self, response: Any) -> bytes:
        """Bound successful and error backend bodies without silently truncating evidence."""
        return self._read_body(response, self.server.max_response_bytes)

    def _request_json(self) -> dict[str, Any]:
        """Read and validate the HTTP request body as a JSON object."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if length <= 0:
            raise ValueError("Request body must be non-empty")
        if length > self.server.max_request_bytes:
            raise ValueError("Gateway request exceeds max_request_bytes")
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Transfer-Encoding is unsupported")
        encoded = self._read_body(self.rfile, self.server.max_request_bytes, length)
        try:
            body = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Request body must be valid JSON") from error
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
        return body

    def _bearer_token(self) -> str:
        authorization = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not authorization.startswith(prefix) or not authorization[len(prefix):]:
            raise ValueError("Codex gateway requires a session bearer token")
        return authorization[len(prefix):]

    def _json(self, status: HTTPStatus, payload: dict[str, Any], model_calls: Optional[int] = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(body) > self.server.max_response_bytes:
            if status < HTTPStatus.BAD_REQUEST:
                raise ValueError("Gateway response exceeds max_response_bytes")
            body = b""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if model_calls is not None:
            self.send_header("X-Model-Calls", str(model_calls))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": {"message": message.replace(self.server.admin_token, "[redacted]")[:2048],
                                      "type": "gateway_error"}})

    # BaseHTTPRequestHandler's first argument is positional; avoid shadowing the format builtin.
    def log_message(self, format_string: str, *args: Any) -> None:  # pylint: disable=arguments-differ
        """Route HTTP server diagnostics through the application logger."""
        logger.debug("Codex gateway: " + format_string, *args)


class _GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: _State, admin_token: str | None,
                 max_request_bytes: int, max_response_bytes: int) -> None:
        """Bind one HTTP server to its shared gateway state."""
        self.state = state
        if admin_token is not None and (not isinstance(admin_token, str) or not admin_token.strip()):
            raise ValueError("Gateway admin_token must be a non-empty string")
        for name, value in (("max_request_bytes", max_request_bytes), ("max_response_bytes", max_response_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Gateway {name} must be a positive integer")
        self.admin_token = admin_token or secrets.token_urlsafe(32)
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        super().__init__(address, _Handler)


class CodexGateway:
    """Own the one protocol adapter used by every Codex episode on a node."""

    def __init__(
        self,
        host: str,
        port: int,
        backend_url: str,
        model_name: str,
        request_timeout: float,
        max_inflight_requests: int = 1,
        admin_token: str | None = None,
        max_request_bytes: int = 8 * 1024 * 1024,
        max_response_bytes: int = 32 * 1024 * 1024,
        inference_only: bool = False,
        backend_key_file: Path | None = None,
    ) -> None:
        """Initialize an unstarted gateway."""
        if not host:
            raise ValueError("Codex gateway host must be non-empty")
        if not 0 <= port < 65536:
            raise ValueError("Codex gateway port must be in [0, 65535]")
        self._server = _GatewayServer(
            (host, port),
            _State(backend_url, model_name, request_timeout, max_inflight_requests,
                   inference_only=inference_only, backend_key_file=backend_key_file),
            admin_token, max_request_bytes, max_response_bytes,
        )
        self._thread: Optional[threading.Thread] = None

    @property
    def admin_token(self) -> str:
        """Return the controller-only management credential; never pass it to candidates."""
        return self._server.admin_token

    @property
    def address(self) -> tuple[str, int]:
        """Return the bound gateway address."""
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        """Start request serving on a daemon thread."""
        if self._thread is not None:
            raise RuntimeError("Codex gateway is already running")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="hyper-rl-codex-gateway",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop request serving and release the listening socket."""
        if self._thread is None:
            self._server.server_close()
            return
        self._server.shutdown()
        try:
            self._server.state.drain()
        finally:
            self._server.server_close()
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("Codex gateway thread did not stop")
        self._thread = None
