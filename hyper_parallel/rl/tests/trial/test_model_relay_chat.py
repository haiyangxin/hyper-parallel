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
"""Local socket contracts for the fixed DeepSeek candidate model route."""

import asyncio
import json
import sys
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from rl.agentic.envs import model_relay


async def _candidate_post(url: str, path: str, body: bytes) -> bytes:
    parsed = urlparse(url)
    reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
    try:
        writer.write((f"POST {path} HTTP/1.0\r\nHost: attacker.invalid\r\n"
                      f"Authorization: Bearer other-session\r\nContent-Length: {len(body)}\r\n\r\n").encode()
                     + body)
        await writer.drain()
        return await asyncio.wait_for(reader.read(), 3)
    finally:
        writer.close()
        await writer.wait_closed()


def _local_endpoint(monkeypatch: pytest.MonkeyPatch, failures: list) -> None:
    original_spawn = asyncio.create_subprocess_exec

    async def local_spawn(*argv: str, **kwargs: object) -> asyncio.subprocess.Process:
        """Run the candidate relay script locally for socket contract tests."""
        script = argv.index("-c")
        return await original_spawn(sys.executable, "-I", "-u", *argv[script:], **kwargs)

    monkeypatch.setattr(model_relay.asyncio, "create_subprocess_exec", local_spawn)
    monkeypatch.setattr(model_relay, "request_gateway_json", lambda *args: failures.append(args) or {})


def test_chat_route_binds_session_and_preserves_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only Chat Completions reaches the fixed gateway with controller credentials."""
    failures = []
    _local_endpoint(monkeypatch, failures)

    async def scenario() -> None:
        """Exercise the relay with an isolated local upstream and close its resources."""
        requests = []
        sse = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'

        async def upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Serve the controlled HTTP response used by this relay contract."""
            headers = await reader.readuntil(b"\r\n\r\n")
            length = int(next(line.split(b":", 1)[1] for line in headers.split(b"\r\n")
                              if line.lower().startswith(b"content-length:")))
            requests.append((headers, await reader.readexactly(length)))
            writer.write(b"HTTP/1.0 200 OK\r\nContent-Type: text/event-stream\r\n\r\n" + sse)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(upstream, "127.0.0.1", 0)
        address = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1"
        relay = model_relay.CandidateRelay(SimpleNamespace(name="unused"), address, "bound-session",
                                          request_timeout=2, max_request_bytes=1024, max_response_bytes=1024,
                                          route="chat_completions")
        try:
            endpoint = await relay.start()
            assert endpoint.endswith("/v1")
            body = json.dumps({"messages": [], "url": "http://attacker.invalid"}).encode()
            for forbidden in ("/v1/responses", "/internal/sessions", "/v1/chat/completions?other=1"):
                response = await _candidate_post(endpoint, forbidden, body)
                assert b" 403 " in response.split(b"\r\n")[0]
            response = await _candidate_post(endpoint, "/v1/chat/completions", body)
            assert b"Content-Type: text/event-stream" in response
            assert response.endswith(sse)
            assert len(requests) == 1
            headers, forwarded = requests[0]
            assert headers.startswith(b"POST /v1/chat/completions HTTP/1.0\r\n")
            assert b"Authorization: Bearer bound-session\r\n" in headers
            assert b"other-session" not in headers and b"attacker.invalid" not in headers
            assert forwarded == body
        finally:
            await relay.close()
            server.close()
            await server.wait_closed()
        assert not failures

    asyncio.run(scenario())


def test_chat_budget_is_delivered_before_controller_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A structured gateway budget response is observed only after candidate delivery."""
    failures = []
    _local_endpoint(monkeypatch, failures)

    async def scenario() -> None:
        """Exercise the relay with an isolated local upstream and close its resources."""
        termination = {"reason": "max_completions", "limit": 2, "completed": 2, "policy_version": 3}
        payload = json.dumps({"error": {"type": "completion_budget_exhausted"},
                              "termination": termination}).encode()

        async def upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Serve the controlled HTTP response used by this relay contract."""
            await reader.readuntil(b"\r\n\r\n")
            await reader.readexactly(2)
            writer.write(b"HTTP/1.0 409 Conflict\r\nContent-Type: application/json\r\n"
                         + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(upstream, "127.0.0.1", 0)
        address = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1"
        relay = model_relay.CandidateRelay(SimpleNamespace(name="unused"), address, "session",
                                          request_timeout=2, route="chat_completions")
        try:
            endpoint = await relay.start()
            response = await _candidate_post(endpoint, "/v1/chat/completions", b"{}")
            assert b" 409 " in response.split(b"\r\n")[0]
            assert response.endswith(payload)
            await asyncio.wait_for(relay.budget_exhausted.wait(), 2)
            assert relay.budget_termination == termination
        finally:
            await relay.close()
            server.close()
            await server.wait_closed()
        assert not failures

    asyncio.run(scenario())


def test_chat_transport_limit_reports_deepseek_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Oversized SSE is not delivered as a successful model action."""
    failures = []
    _local_endpoint(monkeypatch, failures)

    async def scenario() -> None:
        """Exercise the relay with an isolated local upstream and close its resources."""
        async def upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Serve the controlled HTTP response used by this relay contract."""
            await reader.readuntil(b"\r\n\r\n")
            await reader.readexactly(2)
            writer.write(b"HTTP/1.0 200 OK\r\nContent-Type: text/event-stream\r\n\r\n"
                         + b"data: " + b"x" * 128 + b"\n\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(upstream, "127.0.0.1", 0)
        address = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1"
        relay = model_relay.CandidateRelay(SimpleNamespace(name="unused"), address, "session",
                                          request_timeout=2, max_response_bytes=32, route="chat_completions")
        request = None
        try:
            endpoint = await relay.start()
            request = asyncio.create_task(_candidate_post(endpoint, "/v1/chat/completions", b"{}"))
            await asyncio.wait_for(relay._pump_task, 3)
            with pytest.raises(model_relay.RelayError, match="transport failed"):
                await relay.close()
            assert failures[0][0:2] == ("DeepSeek", "POST")
            assert failures[0][3]["reason"] == "Candidate model transport failed"
            assert not relay.budget_exhausted.is_set()
        finally:
            if request is not None:
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_chat_route_rejects_unknown_protocol() -> None:
    """The candidate cannot request arbitrary relay paths."""
    with pytest.raises(ValueError, match="route"):
        model_relay.CandidateRelay(SimpleNamespace(name="unused"), "http://127.0.0.1:1234/v1", "session",
                                   route="/internal/sessions")
