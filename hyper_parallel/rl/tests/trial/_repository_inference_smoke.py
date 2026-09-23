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
"""Real pinned CLI and Docker inference wiring with a scripted, token-free backend."""

from __future__ import annotations

import argparse
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from typing import Any
import urllib.error
import urllib.request

from rl.agentic.codex.gateway import CodexGateway
from rl.agentic.codex.harness import CodexAgentProgram, RepositoryInferenceResult
from examples.code_agent.prepare_data import adapt_row
from examples.code_agent.task import load_fixture, task_manifest


def require(condition: bool, message: str) -> None:
    """Reject a failed functional acceptance condition."""
    if not condition:
        raise RuntimeError(message)


def fixture_command() -> str:
    """Build a deterministic fixture repair solely for synthetic transport validation."""
    fixture, baseline = load_fixture("word_counts")
    lines = ["*** Begin Patch"]
    for name, content in fixture["reference"].items():
        if content is None:
            lines.append(f"*** Delete File: {name}")
        elif name not in baseline:
            lines.append(f"*** Add File: {name}")
            lines.extend("+" + line for line in content.splitlines())
        else:
            lines.extend((f"*** Update File: {name}", "@@"))
            lines.extend("-" + line for line in baseline[name].decode().splitlines())
            lines.extend("+" + line for line in content.splitlines())
    lines.append("*** End Patch")
    # Force a genuine pending session so the next scripted call must poll its real ID.
    return ("set -e\nsleep 1\ncat src/solution.py\n"
            "python /opt/hyper-codex-home/checked_patch.py <<'PATCH'\n"
            + "\n".join(lines) + "\nPATCH\npython public_test.py")


class ScriptedBackend(ThreadingHTTPServer):
    """Own the three-call fixture sequence and immutable wire evidence."""

    def __init__(self) -> None:
        """Bind a local scripted backend and initialize its fixture evidence."""
        super().__init__(("127.0.0.1", 0), BackendHandler)
        self.requests: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.pending_session_id: int | None = None
        self.command = fixture_command()

    def completion(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return ordinary Chat data with no training token or parser evidence."""
        ordinal = len(self.requests)
        self.requests.append(request)
        names = {tool["function"]["name"] for tool in request.get("tools", [])}
        require(names == {"exec_command", "write_stdin"}, "Pinned repository tool pair changed")
        for field in ("logprobs", "top_logprobs", "return_token_ids"):
            require(field not in request, f"Inference requested training-only field {field}")
        outputs = [item.get("content", "") for item in request["messages"] if item.get("role") == "tool"]
        if ordinal == 0:
            name = "exec_command"
            arguments = {"cmd": self.command, "yield_time_ms": 1}
        elif ordinal == 1:
            require(bool(outputs), "Missing real CLI pending feedback")
            matched = re.search(r"Process running with session ID (\d+)", outputs[-1])
            require(matched is not None, "CLI command did not expose a pending session")
            self.pending_session_id = int(matched.group(1))
            name = "write_stdin"
            arguments = {"session_id": self.pending_session_id, "chars": "", "yield_time_ms": 1000}
        elif ordinal == 2:
            require(bool(outputs), "Missing poll result")
            require(all(marker in outputs[-1] for marker in (
                "Process exited with code 0", "PATCH VERIFIED:", "VERIFIED src/", "public checks passed",
            )), "Actual edit/test/exit feedback did not reach the next model request")
            response = {"id": "scripted-final", "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Synthetic fixture repaired and tested."}}]}
            self.responses.append(response)
            return response
        else:
            raise RuntimeError("Unexpected additional model request")
        response = {"id": f"scripted-{ordinal}", "choices": [{"index": 0, "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": [{
                        "id": f"fixture-call-{ordinal}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }]}}]}
        self.responses.append(response)
        return response


class BackendHandler(BaseHTTPRequestHandler):
    """Expose only the scripted Chat route required by the production gateway."""

    server: ScriptedBackend

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Capture a real gateway request and return the next scripted response."""
        try:
            require(self.path == "/v1/chat/completions", "Unexpected backend path")
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            response = self.server.completion(request)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            self.server.errors.append(str(error))
            self.send_error(500, "Synthetic acceptance failed; consult backend evidence")
            return
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format_string: str, *args: Any) -> None:
        """Keep the synthetic HTTP server quiet."""


def parse_args() -> argparse.Namespace:
    """Require an explicit fixed candidate image and unique evidence directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    """Exercise pending output, real editing and independent grading without training data."""
    args.output.mkdir(parents=True, exist_ok=False)
    backend = ScriptedBackend()
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    gateway = CodexGateway("127.0.0.1", 0, f"http://127.0.0.1:{backend.server_port}",
                           "synthetic-api", 30, inference_only=True)
    gateway.start()
    url = f"http://127.0.0.1:{gateway.address[1]}"
    config = {
        "task_factory": "examples.code_agent.task:build_task", "task_config": {},
        "workspace": {"image": args.image, "run_id": args.run_id, "max_concurrent": 1},
        "session_root": str(args.output / "sessions"), "executable": "codex", "version": "0.152.1",
        "model": "synthetic-api", "model_context_window": 40960, "max_turns": 3,
        "timeout_seconds": 120, "request_timeout": 30,
    }
    prompt = adapt_row({"task_id": "repository-v1:word_counts", "ground_truth": task_manifest("word_counts"),
                       "prompt": [{"role": "user", "content": "Read and repair the repository; run public tests."}]},
                      0)
    summary = {"status": "running", "synthetic_model": True, "real_model_requests": 0}
    try:
        program = CodexAgentProgram(prompt, None, 0, url, config, None, admin_url=url,
                                    admin_token=gateway.admin_token, inference_only=True)
        result = await program.run_inference()
        require(isinstance(result, RepositoryInferenceResult), "Inference returned training data")
        require(result.reward.value == 1., "Independent fixture grader did not confirm the repair")
        require(len(backend.requests) == len(backend.responses) == 3, "Expected exec, poll and final calls")
        require(not backend.errors and backend.pending_session_id is not None, "Pending-session loop failed")
        require(result.captured.get("policy_version") is None, "Inference claimed a training policy version")
        completions = result.captured["completions"]
        require(len(completions) == 3, "Incomplete API evidence")
        for record, response in zip(completions, backend.responses):
            require(record["response"] == response, "Gateway changed captured API response evidence")
            require(record["metadata"].get("trainable") is False, "Inference was marked trainable")
        manifest = json.loads((result.artifact_dir / "submission.manifest.json").read_text())
        require(any(manifest["changes"].values()), "No real frozen file modification")
        request = urllib.request.Request(url + f"/internal/sessions/{result.session_id}",
                                         headers={"Authorization": "Bearer " + gateway.admin_token})
        try:
            with urllib.request.urlopen(request, timeout=5):
                raise RuntimeError("Program leaked its Gateway session")
        except urllib.error.HTTPError as error:
            require(error.code == 404, "Unexpected released-session response")
        summary.update(status="passed", cli_version="0.152.1", pending_session_id=backend.pending_session_id,
                       reward=result.reward.value, reward_metadata=result.reward.metadata,
                       artifact_manifest=manifest, captured_calls=3, training_data_created=False,
                       artifact_dir=str(result.artifact_dir), gateway_session_released=True)
    finally:
        gateway.close()
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=5)
        for name, content in (("backend-requests", backend.requests), ("backend-responses", backend.responses),
                              ("backend-errors", backend.errors), ("summary", summary)):
            (args.output / f"{name}.json").write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
