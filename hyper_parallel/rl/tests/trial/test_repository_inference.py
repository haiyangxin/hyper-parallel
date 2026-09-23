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
"""External-API inference stays isolated from training evidence and credentials."""

import asyncio
import base64
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from rl.agentic.codex import gateway, harness
from rl.agentic.codex.protocol import CodexResponsesProtocol
from rl.agentic.core.program_runner import _trace
from rl.agentic.core.types import RewardResult
from rl.agentic.envs.docker_workspace import CommandResult
from rl.agentic.envs import model_relay
from test_gateway_transport import _http, _repository_request


def _answer(arguments: dict | None = None) -> dict:
    """Return an ordinary API answer with no training-only token evidence."""
    message = {"role": "assistant", "content": "done"}
    if arguments is not None:
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "edit-call", "type": "function", "function": {
                "name": "exec_command", "arguments": json.dumps(arguments)}}]}
    return {"id": "api-answer", "model": "external-model", "choices": [{
        "index": 0, "message": message, "finish_reason": "tool_calls" if arguments is not None else "stop"}]}


class _Backend(BaseHTTPRequestHandler):
    """Record only test-owned credentials at a local fake upstream."""

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Return the next scripted response and retain the received test request."""
        length = int(self.headers["Content-Length"])
        self.server.received.append({"authorization": self.headers.get("Authorization"),
                                     "body": json.loads(self.rfile.read(length))})
        status, body = self.server.answers.pop(0)
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args: object) -> None:
        """Keep synthetic backend requests out of the test runner output."""


class TestInferenceGateway(unittest.TestCase):
    """Use real HTTP boundaries without sampling a model or inventing logprobs."""

    def setUp(self) -> None:
        """Start a local fake API and an inference-only recording gateway."""
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        self.root = Path(self.directory)
        self.key = "test-only-external-secret"
        self.key_path = self.root / "backend-key"
        self.key_path.write_text(self.key + "\n")
        self.backend = ThreadingHTTPServer(("127.0.0.1", 0), _Backend)
        self.backend.answers = [(200, _answer())]
        self.backend.received = []
        self.thread = threading.Thread(target=self.backend.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.backend.server_close)
        self.addCleanup(self.backend.shutdown)
        self.artifacts = self.root / "artifacts"
        self.proxy = gateway.CodexGateway(
            "127.0.0.1", 0, f"http://127.0.0.1:{self.backend.server_port}", "external-model", 2,
            inference_only=True, backend_key_file=self.key_path,
        )
        self.proxy.start()
        self.addCleanup(self.proxy.close)
        payload = {"session_id": "s", "policy_version": None, "max_completions": None,
                   "repository_task": True, "artifact_dir": str(self.artifacts), "inference_only": True}
        status, body = _http(self.proxy, "POST", "/internal/sessions", self.proxy.admin_token,
                             json.dumps(payload).encode())
        self.assertEqual(status, 201, body)

    def _request(self) -> tuple[int, dict]:
        return _http(self.proxy, "POST", "/v1/responses", "s", json.dumps(_repository_request()).encode())

    def _snapshot(self) -> dict:
        status, captured = _http(self.proxy, "GET", "/internal/sessions/s", self.proxy.admin_token)
        self.assertEqual(status, 200)
        return captured

    def test_no_evidence_response_is_inference_only_and_key_is_not_captured(self) -> None:
        """API credentials reach the upstream alone; no sampled IDs are fabricated."""
        status, body = self._request()
        self.assertEqual(status, 200, body)
        captured = self._snapshot()
        self.assertIsNone(captured["policy_version"])
        self.assertEqual(len(captured["completions"]), 1)
        record = captured["completions"][0]
        self.assertTrue(record["metadata"]["inference_only"])
        self.assertIs(record["metadata"]["trainable"], False)
        self.assertIsNone(record["metadata"]["policy_version"])
        self.assertNotIn("prompt_token_ids", record["response"])
        self.assertNotIn("token_ids", record["response"]["choices"][0])
        self.assertEqual(self.backend.received[0]["authorization"], "Bearer " + self.key)
        for field in ("logprobs", "top_logprobs", "return_token_ids"):
            self.assertNotIn(field, self.backend.received[0]["body"])
        self.assertNotIn(self.key, json.dumps(captured))
        for path in self.artifacts.rglob("*"):
            if path.is_file():
                self.assertNotIn(self.key.encode(), path.read_bytes(), str(path))
        with self.assertRaisesRegex(ValueError, "cannot train on inference-only"):
            _trace(record, "candidate")

    def test_none_budget_admits_more_than_one_completion_and_cleanup_seals_session(self) -> None:
        """No call limit does not silently become a one-call budget."""
        self.backend.answers = [(200, _answer()), (200, _answer())]
        self.assertEqual(self._request()[0], 200)
        self.assertEqual(self._request()[0], 200)
        self.assertEqual(len(self._snapshot()["completions"]), 2)
        self.assertEqual(_http(self.proxy, "DELETE", "/internal/sessions/s", self.proxy.admin_token)[0], 200)
        self.assertNotEqual(self._request()[0], 200)
        self.assertEqual(len(self.backend.received), 2)

    def test_bad_tool_arguments_are_recorded_and_rejected_before_cli_delivery(self) -> None:
        """A schema failure cannot be declared trainable without parser evidence."""
        self.backend.answers = [(200, _answer({"cmd": "cat", "stdin": "not an exec argument"}))]
        status, body = self._request()
        self.assertEqual(status, 502, body)
        self.assertEqual(len(self.backend.received), 1)
        self.assertNotIn("not an exec argument", json.dumps(body))
        captured = self._snapshot()
        self.assertEqual(len(captured["completions"]), 1)
        self.assertIs(captured["failure"]["trainable"], False)
        self.assertIn("undeclared_tool_argument", captured["failure"]["failure_reason"])

    def test_backend_error_is_not_a_normal_zero_reward(self) -> None:
        """HTTP failure invalidates inference without producing a fake completion."""
        self.backend.answers = [(503, {"error": "upstream unavailable"})]
        self.assertEqual(self._request()[0], 502)
        captured = self._snapshot()
        self.assertEqual(captured["completions"], [])
        self.assertEqual(captured["failure"]["failure_origin"], "infrastructure")
        self.assertIs(captured["failure"]["trainable"], False)

    def test_backend_echoed_secret_is_withheld_from_errors_and_artifacts(self) -> None:
        """Plaintext credentials are redacted while useful JSON diagnostics survive."""
        self.backend.answers = [(401, {"detail": {"error": {
            "code": "invalid_api_key", "message": "Rejected Bearer " + self.key}}})]
        status, body = self._request()
        self.assertEqual(status, 502)
        captured = self._snapshot()
        for result in (body, captured):
            self.assertIn("HTTP 401", json.dumps(result))
            self.assertIn("invalid_api_key", json.dumps(result))
            self.assertIn("Rejected Bearer", json.dumps(result))
            self.assertIn("[redacted JSON; not raw]", json.dumps(result))
        self.assertNotIn(self.key, json.dumps(body))
        self.assertNotIn(self.key, json.dumps(captured))
        for path in self.artifacts.rglob("*"):
            if path.is_file():
                self.assertNotIn(self.key.encode(), path.read_bytes(), str(path))

    def test_unicode_escaped_secret_preserves_context_error_without_leaking(self) -> None:
        """Decode JSON before redacting nested credential values and dictionary keys."""
        escaped_key = "".join(f"\\u{ord(character):04x}" for character in self.key)
        error = {"detail": {"error": {"code": "context_length_exceeded",
                                    "message": "maximum context is 32768; token=" + self.key},
                            "diagnostics": [{self.key: "Bearer " + self.key}]}}
        encoded = json.dumps(error).replace(self.key, escaped_key).encode()
        self.assertNotIn(self.key.encode(), encoded)
        self.backend.answers = [(400, encoded)]
        status, body = self._request()
        self.assertEqual(status, 502)
        captured = self._snapshot()
        for result in (body, captured):
            serialized = json.dumps(result)
            self.assertIn("HTTP 400", serialized)
            self.assertIn("context_length_exceeded", serialized)
            self.assertIn("maximum context is 32768", serialized)
            self.assertIn("[redacted JSON; not raw]", serialized)
            self.assertNotIn(self.key, serialized)
            self.assertNotIn(escaped_key, serialized)
        for path in self.artifacts.rglob("*"):
            if path.is_file():
                contents = path.read_bytes()
                self.assertNotIn(self.key.encode(), contents, str(path))
                self.assertNotIn(escaped_key.encode(), contents, str(path))

    def test_authenticated_non_json_error_is_withheld_with_explicit_reason(self) -> None:
        """Unstructured authenticated error bodies cannot be claimed safely redacted."""
        self.backend.answers = [(503, ("provider failure token=" + self.key).encode())]
        status, body = self._request()
        self.assertEqual(status, 502)
        captured = self._snapshot()
        for result in (body, captured):
            serialized = json.dumps(result)
            self.assertIn("HTTP 503", serialized)
            self.assertIn("non-JSON or cannot reliably redact", serialized)
            self.assertNotIn("provider failure token=", serialized)
            self.assertNotIn(self.key, serialized)

    def test_authenticated_error_diagnostics_remain_bounded_after_redaction(self) -> None:
        """Large provider errors retain an explicit truncation marker and bounded text."""
        body = json.dumps({"error": {"code": "context_length_exceeded", "message": "x" * 10000,
                                    "credential": self.key}}).encode()
        detail = gateway._backend_error_detail(body, self.key)
        self.assertTrue(detail.startswith("[redacted JSON; not raw]"))
        self.assertIn("context_length_exceeded", detail)
        self.assertTrue(detail.endswith(" [truncated]"))
        self.assertLessEqual(len(detail), 4096 + len(" [truncated]"))
        self.assertNotIn(self.key, detail)

    def test_length_truncated_response_is_an_incomplete_protocol_result(self) -> None:
        """An API output ceiling cannot masquerade as a completed model action."""
        response = _answer()
        response["choices"][0]["finish_reason"] = "length"
        self.backend.answers = [(200, response)]
        self.assertEqual(self._request()[0], 502)
        captured = self._snapshot()
        self.assertEqual(len(captured["completions"]), 1)
        self.assertIs(captured["failure"]["trainable"], False)
        self.assertIn("incomplete", captured["failure"]["failure_reason"])

    def test_finite_inference_budget_preserves_structured_termination(self) -> None:
        """An explicit inference budget still stops admission after completed calls."""
        payload = {"session_id": "bounded", "policy_version": None, "max_completions": 1,
                   "repository_task": True, "inference_only": True}
        self.assertEqual(_http(self.proxy, "POST", "/internal/sessions", self.proxy.admin_token,
                               json.dumps(payload).encode())[0], 201)
        body = json.dumps(_repository_request()).encode()
        self.assertEqual(_http(self.proxy, "POST", "/v1/responses", "bounded", body)[0], 200)
        status, response = _http(self.proxy, "POST", "/v1/responses", "bounded", body)
        self.assertEqual(status, 409, response)
        self.assertEqual(len(self.backend.received), 1)

    def test_training_protocol_and_state_keep_strict_defaults(self) -> None:
        """The external inference option cannot weaken existing training defaults."""
        request = CodexResponsesProtocol().transform_request(_repository_request(), "training-model")
        self.assertTrue(request["logprobs"])
        self.assertTrue(request["return_token_ids"])
        state = gateway._State("http://unused", "training-model", 2)
        with self.assertRaises((TypeError, ValueError)):
            state.register("training", {"policy_version": None, "max_completions": None})

    def test_even_complete_token_fields_cannot_relabel_inference_as_training(self) -> None:
        """A synthetic valid training-shaped record remains rejected when marked inference."""
        response = _answer()
        response["prompt_token_ids"] = [1, 2]
        response["choices"][0].update(token_ids=[3], logprobs={"content": [{"token_id": 3, "logprob": -0.2}]})
        record = {"request": {}, "response": response, "metadata": {"inference_only": True}}
        with self.assertRaisesRegex(ValueError, "cannot train on inference-only"):
            _trace(record, "candidate")
        record["metadata"] = {}
        self.assertEqual(_trace(record, "synthetic training control")["response_ids"], [3])

    def test_invalid_message_role_tool_container_and_duplicate_call_ids_are_rejected(self) -> None:
        """Malformed external responses must not become valid executable actions."""
        transformed = CodexResponsesProtocol().transform_request(
            _repository_request(), "external-model", request_training_evidence=False,
        )
        wrong_role = _answer()
        wrong_role["choices"][0]["message"]["role"] = "user"
        wrong_tools = _answer()
        wrong_tools["choices"][0]["message"]["tool_calls"] = {}
        duplicate = _answer({"cmd": "true"})
        calls = duplicate["choices"][0]["message"]["tool_calls"]
        calls.append(dict(calls[0]))
        for response in (wrong_role, wrong_tools, duplicate):
            with self.subTest(response=response):
                outcome = gateway._Handler._inference_outcome(response, transformed, True, "turn")
                self.assertIs(outcome["trainable"], False)
                self.assertIsNotNone(outcome["failure_origin"])


class TestInferenceBudget(unittest.IsolatedAsyncioTestCase):
    """Only a trusted inference relay may accept a budget without a policy version."""

    async def test_inference_budget_requires_mode_and_successful_delivery(self) -> None:
        """An inference terminal retains the existing acknowledgement requirement."""
        termination = {"reason": "max_completions", "limit": 1, "completed": 1, "policy_version": None}
        reply = {"status": 409, "content_type": "application/json", "body": base64.b64encode(json.dumps({
            "error": {"type": "completion_budget_exhausted"}, "termination": termination,
        }).encode()).decode()}
        self.assertIsNone(model_relay._budget_termination(reply))
        self.assertEqual(model_relay._budget_termination(reply, inference_only=True), termination)
        for delivered in (True, False):
            with self.subTest(delivered=delivered):
                relay = model_relay.CandidateRelay(
                    SimpleNamespace(name="unused"), "http://127.0.0.1:9", "bound", inference_only=True,
                )
                lines = iter((b'{"body":"e30="}\n', json.dumps({"delivered": delivered}).encode() + b"\n", b""))
                relay._exchange = AsyncMock(return_value=reply)
                relay._process = SimpleNamespace(
                    stdout=SimpleNamespace(readline=AsyncMock(side_effect=lambda: next(lines))),
                    stdin=SimpleNamespace(write=MagicMock(), drain=AsyncMock()),
                )
                relay._closing = True
                await asyncio.wait_for(relay._pump(), 2)
                self.assertEqual(relay.budget_exhausted.is_set(), delivered)
                if delivered:
                    self.assertEqual(relay.budget_termination, termination)
                    self.assertIsNone(relay._failure)
                else:
                    self.assertIsNone(relay.budget_termination)
                    self.assertIsInstance(relay._failure, model_relay.RelayError)


@asynccontextmanager
async def _slot(*_args: object):
    """Replace only the external workspace admission boundary."""
    yield


class TestInferenceProgram(unittest.IsolatedAsyncioTestCase):
    """Inference reuses repository lifecycle without constructing training rows."""

    def setUp(self) -> None:
        """Replace process boundaries while retaining repository orchestration."""
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        config = {"task_factory": "examples.code_agent.task:build_task", "max_turns": None,
                  "session_root": self.directory, "request_timeout": 2, "model_context_window": 40960,
                  "workspace": {"image": "sha256:" + "a" * 64}, "max_new_tokens": 10}
        self.task = SimpleNamespace(prepare=AsyncMock(), evaluate=AsyncMock(return_value=RewardResult(1., {"ok": 1.})))
        self.workspace = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), close=AsyncMock(), copy_in=AsyncMock(),
                                         exec=AsyncMock(return_value=CommandResult(0, "", "")),
                                         export=AsyncMock(return_value=Path("frozen.tar")))
        self.relay = SimpleNamespace(start=AsyncMock(return_value="http://127.0.0.1:42/v1"), close=AsyncMock(),
                                     budget_exhausted=asyncio.Event(), budget_termination=None)
        self.capture = {"policy_version": None, "inference_only": True,
                        "completions": [{"ordinal": 0, "request": {}, "response": _answer(),
                        "metadata": {"inference_only": True, "trainable": False, "policy_version": None}}]}
        self.http = MagicMock(side_effect=lambda method, *_args, **_kwargs: self.capture if method == "GET" else {})
        for target, value in (("load_reward_callable", MagicMock(return_value=MagicMock(return_value=self.task))),
                              ("DockerWorkspace", MagicMock(return_value=self.workspace)),
                              ("CandidateRelay", MagicMock(return_value=self.relay)),
                              ("_http_json", self.http), ("acquire_workspace_slot", _slot)):
            patcher = patch.object(harness, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.program = harness.CodexAgentProgram(
            SimpleNamespace(prompt_id="repo", messages=[SimpleNamespace(content="fix")]), None, 0,
            "http://model/v1", config, None, admin_url="http://controller", admin_token="controller-only",
            inference_only=True,
        )
        self.program._validate_container_version = AsyncMock()
        self.program._run_container_codex = AsyncMock(return_value=("done", []))
        self.program._build_rows = MagicMock(side_effect=AssertionError("Inference must not build training rows"))

    async def test_run_inference_grades_real_frozen_artifact_without_rows(self) -> None:
        """A successful inference result contains the independently assigned reward."""
        result = await self.program.run_inference()
        self.assertEqual(result.final_answer, "done")
        self.assertEqual(result.reward.value, 1.)
        self.task.evaluate.assert_awaited_once_with(Path("frozen.tar"))
        self.program._build_rows.assert_not_called()
        self.workspace.close.assert_awaited_once()
        self.relay.close.assert_awaited()
        self.assertEqual(self.http.call_args.args[0], "DELETE")

    async def test_training_run_explicitly_rejects_inference_program(self) -> None:
        """Training entry points reject inference programs before side effects."""
        with self.assertRaisesRegex(ValueError, "inference|train"):
            await self.program.run()
        self.workspace.start.assert_not_awaited()

    async def test_cli_timeout_propagates_and_closes_without_grading(self) -> None:
        """A timed-out CLI is infrastructure failure rather than a zero reward."""
        self.program._run_container_codex.side_effect = TimeoutError("model transport timeout")
        with self.assertRaises(TimeoutError):
            await self.program.run_inference()
        self.task.evaluate.assert_not_awaited()
        self.workspace.close.assert_awaited_once()
        self.relay.close.assert_awaited()
        self.assertEqual(self.http.call_args.args[0], "DELETE")

    async def test_grader_failure_propagates_and_still_cleans_up(self) -> None:
        """Grading exceptions preserve their identity and release owned resources."""
        self.task.evaluate.side_effect = RuntimeError("grader unavailable")
        with self.assertRaisesRegex(RuntimeError, "grader unavailable"):
            await self.program.run_inference()
        self.workspace.close.assert_awaited_once()
        self.relay.close.assert_awaited()
        self.program._build_rows.assert_not_called()

    async def test_cancel_prepare_releases_session_without_fabricating_result(self) -> None:
        """Cancellation during setup cannot produce a successful inference result."""
        self.task.prepare.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.program.run_inference()
        self.workspace.close.assert_awaited_once()
        self.task.evaluate.assert_not_awaited()
        self.assertEqual(self.http.call_args.args[0], "DELETE")
