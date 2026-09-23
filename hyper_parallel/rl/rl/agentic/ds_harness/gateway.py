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
"""Independent DeepSeek Chat Completions recorder for the shared vLLM server."""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import logging
import secrets
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from rl.tool_protocol import inspect_tool_response

logger = logging.getLogger(__name__)


class _CallBudgetExceeded(ValueError):
    """Identify exhaustion of the gateway's explicit model-call budget."""


class CompletionBudgetExhausted(ValueError):
    """Identify a completed repository call budget without marking model failure."""

    def __init__(self, termination: dict[str, Any]) -> None:
        """Retain the verified terminal call count for the candidate relay."""
        self.termination = dict(termination)
        super().__init__(f"DeepSeek repository reached max_completions={termination['limit']}")


class _ToolResponseFailure(RuntimeError):
    """Preserve the original attribution while terminating one DeepSeek model call."""

    def __init__(self, outcome: dict[str, Any]) -> None:
        """Retain evidence-based failure fields for the session trace."""
        super().__init__(outcome["failure_reason"])
        self.outcome = outcome


def _repository_tool_argument_failure(response: dict[str, Any], request: dict[str, Any]) -> dict | None:
    """Reject sampled arguments the SDK cannot execute under the declared schema."""
    declared = {tool["function"]["name"]: tool["function"]["parameters"] for tool in request.get("tools", [])}
    calls = response["choices"][0]["message"].get("tool_calls") or []
    for call in calls:
        function = call["function"]
        name = function["name"]
        if name not in declared:
            return {"failure_origin": "model", "failure_reason": "undeclared_tool_name", "trainable": True,
                    "tool_name": name}
        try:
            arguments = json.loads(function["arguments"])
        except (TypeError, ValueError):
            return {"failure_origin": "model", "failure_reason": "invalid_tool_arguments", "trainable": True,
                    "tool_name": name}
        if not isinstance(arguments, dict):
            return {"failure_origin": "model", "failure_reason": "invalid_tool_arguments", "trainable": True,
                    "tool_name": name}
        parameters = declared[name]
        properties = parameters.get("properties", {})
        required = parameters.get("required") or []
        extra = sorted(set(arguments) - set(properties))
        missing = sorted(set(required) - set(arguments))
        if extra or missing:
            return {"failure_origin": "model", "failure_reason": "invalid_tool_schema", "trainable": True,
                    "tool_name": name, "extra_arguments": extra, "missing_arguments": missing}
        for key, value in arguments.items():
            expected = properties[key].get("type")
            valid = ((expected == "string" and isinstance(value, str))
                     or (expected == "boolean" and isinstance(value, bool))
                     or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
                     or (expected == "number" and isinstance(value, (int, float)) and not isinstance(value, bool)))
            if not valid:
                return {"failure_origin": "model", "failure_reason": "invalid_tool_schema", "trainable": True,
                        "tool_name": name, "invalid_argument": key}
    return None


class DeepSeekChatProtocol:
    """Preserve DeepSeek chat/tool semantics while collecting exact token evidence."""

    def transform_request(self, body: dict[str, Any], served_model: str) -> dict[str, Any]:
        """Convert a streaming Harness request into one non-streaming vLLM request."""
        if not isinstance(body, dict):
            raise ValueError("DeepSeek request must be a JSON object")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("DeepSeek request requires a non-empty messages list")
        request = dict(body)
        request.update(
            {
                "model": served_model,
                "stream": False,
                "logprobs": True,
                "top_logprobs": 0,
                "return_token_ids": True,
            }
        )
        request.pop("stream_options", None)
        effort = request.pop("reasoning_effort", None)
        if effort in {"off", "high", "max"}:
            chat_template_kwargs = request.get("chat_template_kwargs", {})
            if not isinstance(chat_template_kwargs, dict):
                raise ValueError("DeepSeek chat_template_kwargs must be a mapping")
            request["chat_template_kwargs"] = {
                **chat_template_kwargs,
                "enable_thinking": effort != "off",
            }
        elif effort is not None:
            raise ValueError(f"Unsupported DeepSeek reasoning_effort: {effort}")
        return request

    def stream_events(self, response: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Synthesize the OpenAI-compatible SSE chunks consumed by DeepSeek Harness."""
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("vLLM response omitted its first choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("vLLM response omitted its assistant message")
        response_id = str(response.get("id") or f"chatcmpl-{uuid.uuid4().hex}")
        model = str(response.get("model") or "policy")
        created = int(response.get("created") or time.time())
        delta: dict[str, Any] = {"role": "assistant"}
        for name in ("content", "reasoning_content"):
            value = message.get(name)
            if value is not None:
                delta[name] = value
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            delta["tool_calls"] = [
                {**tool_call, "index": index} if isinstance(tool_call, dict) else tool_call
                for index, tool_call in enumerate(tool_calls)
            ]
        yield self._chunk(response_id, model, created, delta, None)
        finish_reason = choice.get("finish_reason")
        if finish_reason is None:
            finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
        final = self._chunk(response_id, model, created, {}, str(finish_reason))
        usage = response.get("usage")
        if isinstance(usage, dict):
            final["usage"] = dict(usage)
        yield final

    @staticmethod
    def _chunk(
        response_id: str,
        model: str,
        created: int,
        delta: dict[str, Any],
        finish_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }


@dataclass
class _Session:
    policy_version: int
    artifact_dir: Path | None
    max_completions: int
    generation: dict[str, Any]
    repository_task: bool = False
    completions: list[dict[str, Any]] = field(default_factory=list)
    failure: dict[str, Any] | None = None
    termination: dict[str, Any] | None = None
    closing: bool = False
    request_lock: Any = field(default_factory=threading.RLock)


class _State:
    """Own registered gateway sessions and captured completion events."""
    def __init__(
        self, backend_url: str, model_name: str, request_timeout: float
    ) -> None:
        """Initialize independent DeepSeek protocol and per-session trace storage."""
        self.backend_url = backend_url.rstrip("/")
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.protocol = DeepSeekChatProtocol()
        self.lock = threading.RLock()
        self.sessions: dict[str, _Session] = {}
        self.released_sessions: set[str] = set()

    def register(self, session_id: str, payload: dict[str, Any]) -> None:
        """Validate and register a new gateway session and its artifact directory."""
        if not session_id:
            raise ValueError("DeepSeek session ID must be non-empty")
        artifact_value = payload.get("artifact_dir")
        artifact_dir = (
            Path(artifact_value).resolve() if isinstance(artifact_value, str) else None
        )
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
        max_completions = int(payload.get("max_completions", 0))
        if max_completions <= 0:
            raise ValueError("DeepSeek session requires positive max_completions")
        generation = payload.get("generation", {})
        if not isinstance(generation, Mapping):
            raise ValueError("DeepSeek session generation settings must be a mapping")
        allowed = {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "ignore_eos",
            "reasoning_effort",
        }
        unknown = set(generation) - allowed
        if unknown:
            raise ValueError(f"Unknown DeepSeek generation settings: {sorted(unknown)}")
        reasoning_effort = generation.get("reasoning_effort")
        if reasoning_effort not in {None, "off", "high", "max"}:
            raise ValueError(
                "DeepSeek reasoning_effort must be 'off', 'high', or 'max'"
            )
        repository_task = payload.get("repository_task", False)
        if not isinstance(repository_task, bool):
            raise ValueError("DeepSeek repository_task must be a boolean")
        session = _Session(
            policy_version=int(payload["policy_version"]),
            artifact_dir=artifact_dir,
            max_completions=max_completions,
            generation=dict(generation),
            repository_task=repository_task,
        )
        with self.lock:
            if session_id in self.released_sessions:
                raise ValueError("Session was already released")
            if session_id in self.sessions:
                raise ValueError(f"DeepSeek session already exists: {session_id}")
            self.sessions[session_id] = session
        self.event(session_id, "session.registered", payload)

    def get(self, session_id: str) -> _Session:
        """Resolve an existing session without creating implicit request state."""
        with self.lock:
            try:
                return self.sessions[session_id]
            except KeyError as error:
                raise ValueError(f"Unknown DeepSeek session: {session_id}") from error

    def remove(self, session_id: str) -> None:
        """Wait for this session's current request before releasing its trace."""
        with self.lock:
            self.released_sessions.add(session_id)
            session = self.sessions.get(session_id)
        if session is None:
            return
        with session.request_lock:
            session.closing = True
            self.event(session_id, "session.released", {})
            with self.lock:
                self.sessions.pop(session_id, None)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        """Seal final admission and include any completed in-flight call and failure."""
        session = self.get(session_id)
        with session.request_lock:
            session.closing = True
            snapshot = {"policy_version": session.policy_version, "completions": list(session.completions),
                        "failure": session.failure}
            if session.repository_task:
                snapshot["termination"] = session.termination
            return snapshot

    def record_failure(self, session: _Session, outcome: dict[str, Any]) -> None:
        """Preserve untrainable failures regardless of subsequent model-budget errors."""
        with self.lock:
            if session.failure is None or (session.failure.get("trainable") is True
                                           and outcome.get("trainable") is not True):
                session.failure = dict(outcome)

    def save_completion(self, session_id: str, record: dict[str, Any]) -> None:
        """Capture a raw response with a contiguous completion ordinal."""
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
        with (
            self.lock,
            (session.artifact_dir / "gateway-events.jsonl").open(
                "a", encoding="utf-8"
            ) as stream,
        ):
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")


class _Handler(BaseHTTPRequestHandler):
    """Translate HTTP gateway requests into versioned backend completions."""
    server: _GatewayServer

    def setup(self) -> None:
        """Bound client I/O so stalled request bodies cannot hold session locks forever."""
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
            session_id = unquote(path[len(prefix) :])
            try:
                snapshot = self.server.state.snapshot(session_id)
            except ValueError as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
                return
            try:
                self._json(HTTPStatus.OK, snapshot)
            except ValueError as error:
                self._error(HTTPStatus.BAD_GATEWAY, str(error))
            return
        self._error(HTTPStatus.NOT_FOUND, "Unknown DeepSeek gateway route")

    def do_POST(self) -> None:  # pylint: disable=C0103
        """Register a session or proxy one DeepSeek chat request."""
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
        if path.rstrip("/") not in {"/chat/completions", "/v1/chat/completions"}:
            self._error(HTTPStatus.NOT_FOUND, "Unknown DeepSeek gateway route")
            return
        try:
            self._proxy_chat()
        except (RuntimeError, ValueError, OSError) as error:
            logger.exception("DeepSeek gateway request failed")
            self._error(HTTPStatus.BAD_GATEWAY, str(error))

    def do_DELETE(self) -> None:  # pylint: disable=C0103
        """Release one captured session while retaining its artifacts."""
        path = urlparse(self.path).path
        if path.startswith("/internal/") and not self._authorize_admin():
            return
        prefix = "/internal/sessions/"
        if not path.startswith(prefix):
            self._error(HTTPStatus.NOT_FOUND, "Unknown DeepSeek gateway route")
            return
        session_id = unquote(path[len(prefix) :])
        try:
            self.server.state.remove(session_id)
        except ValueError as error:
            self._error(HTTPStatus.NOT_FOUND, str(error))
            return
        self._json(HTTPStatus.OK, {"released": session_id})

    def _proxy_chat(self, original: dict[str, Any] | None = None) -> None:
        """Forward a versioned chat request and capture its completion."""
        session_id = self.headers.get("x-deepseek-harness-session-id", "")
        if not session_id:
            session_id = self._bearer_token()
        session = self.server.state.get(session_id)
        with session.request_lock:
            if session.closing:
                raise ValueError(f"DeepSeek session is sealed: {session_id}")
            try:
                if original is None:
                    original = self._request_json()
                self._complete_chat(session_id, session, original)
            except _CallBudgetExceeded:
                self.server.state.record_failure(session, {
                    "failure_origin": "model", "failure_reason": "call_budget_exhausted", "trainable": True,
                })
                raise
            except CompletionBudgetExhausted as error:
                try:
                    payload = {
                        "error": {"type": "completion_budget_exhausted", "message": str(error)},
                        "termination": error.termination,
                    }
                    if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > self.server.max_response_bytes:
                        raise ValueError("DeepSeek budget response exceeds max_response_bytes")
                    self._json(HTTPStatus.CONFLICT, payload)
                except (RuntimeError, ValueError, OSError) as delivery_error:
                    self.server.state.record_failure(session, {
                        "failure_origin": "infrastructure", "failure_reason": str(delivery_error), "trainable": False,
                    })
                    raise
            except _ToolResponseFailure as error:
                self.server.state.record_failure(session, error.outcome)
                raise
            except (RuntimeError, ValueError, OSError) as error:
                self.server.state.record_failure(session, {
                    "failure_origin": "infrastructure", "failure_reason": str(error), "trainable": False,
                })
                raise

    def _complete_chat(self, session_id: str, session: _Session, original: dict[str, Any]) -> None:
        """Keep budget checking, backend execution and evidence capture atomic per session."""
        if session.failure is not None:
            raise RuntimeError("DeepSeek session has an unresolved request failure")
        if session.termination is not None:
            raise CompletionBudgetExhausted(session.termination)
        if len(session.completions) >= session.max_completions:
            if session.repository_task:
                session.termination = {
                    "reason": "max_completions", "limit": session.max_completions,
                    "completed": len(session.completions), "policy_version": session.policy_version,
                }
                self.server.state.event(session_id, "session.budget_exhausted", session.termination)
                raise CompletionBudgetExhausted(session.termination)
            raise _CallBudgetExceeded(
                f"DeepSeek session exceeded max_completions={session.max_completions}"
            )
        protocol_request = dict(original)
        reasoning_effort = session.generation.get("reasoning_effort")
        if reasoning_effort is not None:
            protocol_request["reasoning_effort"] = reasoning_effort
        transformed = self.server.state.protocol.transform_request(
            protocol_request, self.server.state.model_name
        )
        if session.repository_task:
            # transform_request makes a shallow copy; never rewrite the SDK's raw request.
            transformed["tools"] = copy.deepcopy(transformed.get("tools", []))
            for tool in transformed.get("tools", []):
                function = tool.get("function", {})
                parameters = function.get("parameters", {})
                if not isinstance(parameters.get("properties"), dict):
                    raise ValueError("Repository DeepSeek tool schema lacks argument properties")
                function["parameters"] = {**parameters, "additionalProperties": False}
        transformed.update(
            {
                name: value
                for name, value in session.generation.items()
                if name != "reasoning_effort"
            }
        )
        try:
            response = self._backend_request(transformed)
        except (RuntimeError, ValueError, OSError) as error:
            self.server.state.event(session_id, "completion.backend_rejected",
                                    self._rejected_request_evidence(original, transformed, error))
            raise
        outcome = inspect_tool_response(response)
        if session.repository_task and outcome["failure_origin"] is None:
            tool_failure = _repository_tool_argument_failure(response, transformed)
            if tool_failure is not None:
                outcome = tool_failure
        self.server.state.save_completion(
            session_id,
            {
                "timestamp": time.time(),
                "original_request": original,
                "request": transformed,
                "response": response,
                "metadata": {
                    "policy_version": session.policy_version,
                    **outcome,
                },
            },
        )
        if outcome["failure_origin"] is not None:
            raise _ToolResponseFailure(outcome)
        if bool(original.get("stream")):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self._write_sse(response)
            return
        self._json(HTTPStatus.OK, response)

    @staticmethod
    def _rejected_request_evidence(original: dict[str, Any], transformed: dict[str, Any],
                                   error: Exception) -> dict[str, Any]:
        """Keep the rejected model context when bounded, and hashes otherwise."""
        evidence: dict[str, Any] = {"error_type": type(error).__name__}
        for name, request in (("original", original), ("model", transformed)):
            encoded = json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
            evidence[f"{name}_bytes"] = len(encoded)
            evidence[f"{name}_sha256"] = hashlib.sha256(encoded).hexdigest()
            if len(encoded) <= 256 * 1024:
                evidence[f"{name}_request"] = request
        return evidence

    def _write_sse(self, response: dict[str, Any]) -> None:
        deadline = time.monotonic() + self.server.state.request_timeout
        total = len(b"data: [DONE]\n\n")
        for event in self.server.state.protocol.stream_events(response):
            if time.monotonic() >= deadline:
                raise TimeoutError("Gateway SSE write exceeded request_timeout")
            payload = json.dumps(event, ensure_ascii=False).encode("utf-8")
            frame = b"data: " + payload + b"\n\n"
            total += len(frame)
            if total > self.server.max_response_bytes:
                raise ValueError("Gateway SSE response exceeds max_response_bytes")
            self.wfile.write(frame)
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _backend_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON chat request to the configured inference backend."""
        data = json.dumps(payload).encode("utf-8")
        if len(data) > self.server.max_request_bytes:
            raise ValueError("Backend request exceeds max_request_bytes")
        request = urllib.request.Request(
            f"{self.server.state.backend_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.server.state.request_timeout
            ) as response:
                body = self._read_backend(response)
        except urllib.error.HTTPError as error:
            detail = self._read_backend(error).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM chat completion failed with HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"vLLM chat completion request failed: {error.reason}"
            ) from error
        except http.client.RemoteDisconnected as error:
            raise RuntimeError(
                "vLLM chat completion connection closed before a response"
            ) from error
        try:
            decoded = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("vLLM chat completion returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise RuntimeError("vLLM chat completion returned a non-object response")
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
        if not authorization.startswith(prefix) or not authorization[len(prefix) :]:
            raise ValueError("DeepSeek gateway requires a session identity")
        return authorization[len(prefix) :]

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(body) > self.server.max_response_bytes:
            if status < HTTPStatus.BAD_REQUEST:
                raise ValueError("Gateway response exceeds max_response_bytes")
            body = b""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": {"message": message.replace(self.server.admin_token, "[redacted]")[:2048],
                                      "type": "gateway_error"}})

    # BaseHTTPRequestHandler's first argument is positional; avoid shadowing the format builtin.
    def log_message(self, format_string: str, *args: Any) -> None:  # pylint: disable=arguments-differ
        """Route HTTP diagnostics through the application logger."""
        logger.debug("DeepSeek gateway: " + format_string, *args)


class _GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: _State, admin_token: str | None,
                 max_request_bytes: int, max_response_bytes: int) -> None:
        """Bind the request handler to this gateway's protocol and sessions."""
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


class DeepSeekGateway:
    """Own the protocol adapter used only by DeepSeek Harness episodes."""

    def __init__(
        self,
        host: str,
        port: int,
        backend_url: str,
        model_name: str,
        request_timeout: float,
        admin_token: str | None = None,
        max_request_bytes: int = 8 * 1024 * 1024,
        max_response_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        """Initialize an unstarted DeepSeek gateway."""
        if not host:
            raise ValueError("DeepSeek gateway host must be non-empty")
        if not 0 <= port < 65536:
            raise ValueError("DeepSeek gateway port must be in [0, 65535]")
        self._server = _GatewayServer(
            (host, port), _State(backend_url, model_name, request_timeout),
            admin_token, max_request_bytes, max_response_bytes,
        )
        self._thread: threading.Thread | None = None

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
            raise RuntimeError("DeepSeek gateway is already running")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="hyper-rl-deepseek-gateway",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop request serving and release the listening socket."""
        if self._thread is None:
            self._server.server_close()
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("DeepSeek gateway thread did not stop")
        self._thread = None
