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
"""Explicit CPU DS SDK/Gateway smoke for a candidate's model-visible bash output cap."""

import argparse
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig

from rl.agentic.core.program_runner import request_gateway_json
from rl.agentic.ds_harness.gateway import DeepSeekGateway
from rl.agentic.envs.docker_workspace import DockerWorkspace
from rl.agentic.envs.model_relay import CandidateRelay


_COMMAND = "python -c 'import sys; sys.stdout.write(\"A\"*100000); sys.stderr.write(\"B\"*100000)'"


class _Backend(BaseHTTPRequestHandler):
    """Return one bash action and then a final answer with synthetic token evidence."""

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Emit a long-output action followed by a terminal response."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        ordinal = len(self.server.requests) - 1
        tool_calls = []
        if ordinal == 0:
            arguments = {"command": _COMMAND, "description": "Print oversized stdout and stderr streams"}
            tool_calls = [{"id": "long-output", "type": "function", "function": {
                "name": "bash", "arguments": json.dumps(arguments),
            }}]
            raw = "<tool_call>" + json.dumps({"name": "bash", "arguments": arguments}) + "</tool_call>"
        else:
            raw = "The long command completed."
        action_ids = [101 + ordinal * 2, 102 + ordinal * 2]
        response = {"id": f"synthetic-long-{ordinal}", "model": "transport-test", "created": 1,
                    "prompt_token_ids": [10, 20 + ordinal],
                    "choices": [{"index": 0, "token_ids": action_ids,
                                 "finish_reason": "tool_calls" if tool_calls else "stop",
                                 "message": {"role": "assistant", "content": None if tool_calls else raw,
                                             "tool_calls": tool_calls},
                                 "logprobs": {"content": [{"token_id": token, "token": "fixture", "logprob": -0.2}
                                                          for token in action_ids]}}],
                    "hyper_tool_protocol": [{"parser_input": raw, "engine_text": raw, "decoded_tokens": raw,
                                             "token_ids": action_ids,
                                             "parser_result": {"tool_calls": tool_calls}}]}
        encoded = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args: object) -> None:
        """Keep synthetic model server logs quiet."""


def _sdk_turn(workspace_name: str, model_url: str, session_id: str) -> str:
    launch_args = ("docker", "exec", "-i", "-w", "/workspace",
                   "-e", "DSH_HOME=/opt/hyper-codex-home",
                   "-e", "DSH_CWD=/workspace",
                   "-e", f"DSH_SESSION_ROOT=/tmp/hyper-dsh-{session_id}",
                   "-e", "DSH_CORDIS_CONFIG=/opt/deepseek-harness/cordis.yml",
                   "-e", "DSH_TELEMETRY_DISABLED=1",
                   "-e", f"DEEPSEEK_BASE_URL={model_url}",
                   "-e", f"DEEPSEEK_API_KEY={session_id}",
                   "-e", "NO_PROXY=127.0.0.1,localhost",
                   workspace_name, "/opt/deepseek-harness/runtime/dsh-jsonrpc-agent-pkg-linux-arm64")
    config = DeepSeekHarnessConfig(
        provider="deepseek-official", model="transport-test", max_tokens=256,
        cwd="/workspace", runtime_cwd="/tmp", launch_args_override=launch_args,
        base_url=model_url, api_key=session_id, request_timeout_seconds=90,
        shutdown_timeout_seconds=3,
    )
    with DeepSeekHarness(config) as harness:
        return harness.run("Use bash to print the requested long output, then finish.",
                           session_id=session_id).final_response


async def run(args: argparse.Namespace) -> None:
    """Capture both actual model requests without grading a candidate artifact."""
    args.output.mkdir(parents=True, exist_ok=False)
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _Backend)
    backend.requests = []
    worker = threading.Thread(target=backend.serve_forever, daemon=True)
    worker.start()
    gateway = DeepSeekGateway("127.0.0.1", 0, f"http://127.0.0.1:{backend.server_port}",
                              "transport-test", 90)
    gateway.start()
    admin_url = f"http://127.0.0.1:{gateway.address[1]}"
    session_id = "ds-bash-output-budget"
    workspace = DockerWorkspace(args.image, network="none", name_prefix="hyper-ds-bash-output", run_id=args.run_id)
    relay = CandidateRelay(workspace, admin_url + "/v1", session_id, request_timeout=90,
                           route="chat_completions", admin_url=admin_url, admin_token=gateway.admin_token)
    try:
        await asyncio.to_thread(request_gateway_json, "DeepSeek", "POST", admin_url + "/internal/sessions",
                                {"session_id": session_id, "policy_version": 7,
                                 "artifact_dir": str(args.output), "max_completions": 3,
                                 "generation": {"max_tokens": 256, "reasoning_effort": "off"}},
                                30, gateway.admin_token)
        await workspace.start()
        model_url = await relay.start()
        answer = await asyncio.to_thread(_sdk_turn, workspace.name, model_url, session_id)
        await relay.close()
        capture = await asyncio.to_thread(request_gateway_json, "DeepSeek", "GET",
                                          admin_url + f"/internal/sessions/{session_id}", None, 30,
                                          gateway.admin_token)
        requests = backend.requests
        if answer.strip() != "The long command completed." or len(requests) != 2 or len(capture["completions"]) != 2:
            raise RuntimeError("DeepSeek long-output SDK/Gateway turn was incomplete")
        first_tools = [tool["function"]["name"] for tool in requests[0]["tools"]]
        second_tools = [tool["function"]["name"] for tool in requests[1]["tools"]]
        tool_messages = [message for message in requests[1]["messages"] if message.get("role") == "tool"]
        if first_tools != ["bash"] or second_tools != ["bash"] or len(tool_messages) != 1:
            raise RuntimeError("Unexpected model-visible tools or tool feedback")
        feedback = str(tool_messages[0]["content"])
        evidence = {"image": args.image, "requests": len(requests), "tools": [first_tools, second_tools],
                    "tool_feedback_bytes": len(feedback.encode()), "stdout_tail_chars": feedback.count("A"),
                    "stderr_tail_chars": feedback.count("B"), "feedback_tail": feedback[-600:],
                    "gateway_failure": capture["failure"], "answer": answer}
        (args.output / "audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
        if (not 0 < evidence["tool_feedback_bytes"] <= 9000 or evidence["stdout_tail_chars"] < 3900
                or evidence["stderr_tail_chars"] < 3900 or capture["failure"] is not None):
            raise RuntimeError(f"DeepSeek output cap did not reach the model as expected: {evidence}")
        print(json.dumps(evidence))
    finally:
        await relay.close()
        await workspace.close()
        await asyncio.to_thread(gateway.close)
        backend.shutdown()
        worker.join(timeout=3)
        backend.server_close()


def main() -> None:
    """Parse the pinned image and output location, then run the output-cap probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
