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
"""Exercise the pinned DeepSeek binary and Cordis stack without model hardware."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig


_REQUESTS = []
_RUNTIME = "/opt/deepseek-harness/runtime/dsh-jsonrpc-agent-pkg-linux-arm64"
_CORDIS = "/opt/deepseek-harness/cordis.yml"


class _Model(BaseHTTPRequestHandler):
    """Reply to a real SDK request with a deterministic Chat Completions stream."""

    def do_POST(self) -> None:
        """Record the SDK request and stream the fixed smoke response."""
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _REQUESTS.append((self.path, json.loads(body)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for delta, reason in (({"role": "assistant", "content": "SMOKE_OK"}, None), ({}, "stop")):
            payload = {"id": "chatcmpl-smoke", "object": "chat.completion.chunk", "created": int(time.time()),
                       "model": "qwen3", "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, _format: str, *_args: object) -> None:
        """Keep the smoke result limited to contract evidence."""


def main() -> None:
    """Require an actual SDK turn and model request from the isolated candidate."""
    if not Path(_RUNTIME).is_file() or not Path(_CORDIS).is_file():
        raise RuntimeError("Pinned DeepSeek runtime files are missing")
    server = HTTPServer(("127.0.0.1", 0), _Model)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        config = DeepSeekHarnessConfig(
            provider="deepseek-official", model="qwen3", max_tokens=64,
            cwd="/workspace", runtime_cwd="/workspace", session_root="/tmp/dsh-smoke-sessions",
            cordis=_CORDIS, runtime_bin=_RUNTIME,
            base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="smoke",
            request_timeout_seconds=30, shutdown_timeout_seconds=3,
            env={"DSH_HOME": "/opt/hyper-agent-home", "DSH_TELEMETRY_DISABLED": "1",
                 "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"},
        )
        with DeepSeekHarness(config) as harness:
            result = harness.run("Reply exactly SMOKE_OK.", session_id="smoke")
        if result.final_response.strip() != "SMOKE_OK":
            raise RuntimeError(f"Unexpected DeepSeek final response: {result.final_response!r}")
        if len(_REQUESTS) != 1 or _REQUESTS[0][0] != "/v1/chat/completions":
            raise RuntimeError(f"Unexpected DeepSeek model requests: {_REQUESTS!r}")
        tools = [item.get("function", {}).get("name") for item in _REQUESTS[0][1].get("tools", [])]
        bash_schema = next(item["function"]["parameters"] for item in _REQUESTS[0][1]["tools"]
                           if item.get("function", {}).get("name") == "bash")
        print(json.dumps({"status": "passed", "requests": len(_REQUESTS),
                          "path": _REQUESTS[0][0], "tools": tools, "bash_schema": bash_schema,
                          "finish_reason": result.finish_reason}))
    finally:
        server.shutdown()
        worker.join(timeout=3)


if __name__ == "__main__":
    main()
