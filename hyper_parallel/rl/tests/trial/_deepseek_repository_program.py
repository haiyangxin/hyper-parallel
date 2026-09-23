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
"""Real DS runtime, relay, candidate and grader with scripted model actions."""

import argparse
import asyncio
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import threading
from typing import Any

import torch

from rl.agentic.ds_harness.gateway import DeepSeekGateway
from rl.agentic.ds_harness.harness import DeepSeekAgentProgram
from examples.code_agent.prepare_data import adapt_row
from examples.code_agent.task import load_fixture, task_manifest


class Backend(BaseHTTPRequestHandler):
    """Supply one real bash action and one final response with exact token evidence."""

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Return scripted repository actions and matching synthetic token evidence."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        ordinal = len(self.server.requests) - 1
        calls = []
        last_message = body.get("messages", [])[-1]
        if (self.server.compaction_check
                and "You are now acting as a compaction engine" in str(last_message.get("content", ""))):
            self.server.compaction_calls.append(ordinal)
            raw = ("The repository task is still in progress. Earlier bash calls inspected source and printed "
                   "bounded diagnostic output. Continue using bash; make only verified edits and run tests.")
        elif self.server.compaction_check and self.server.normal_calls >= 18:
            raw = "The source inspection is complete; no repository edit was made."
        elif ordinal == 0 or self.server.compaction_check:
            tools = [tool["function"] for tool in body.get("tools", [])]
            selected = next((tool for tool in tools if tool["name"] == "bash"), None)
            if selected is None:
                raise RuntimeError(f"DS did not expose bash: {[tool['name'] for tool in tools]}")
            if self.server.compaction_check:
                self.server.normal_calls += 1
            arguments = {"command": self.server.command, "description": "Edit source and run public test"}
            calls = [{"id": "scripted-bash", "type": "function", "function": {
                "name": "bash", "arguments": json.dumps(arguments),
            }}]
            raw = "<tool_call>" + json.dumps({"name": "bash", "arguments": arguments}) + "</tool_call>"
        else:
            tool_messages = [message for message in body["messages"] if message.get("role") == "tool"]
            feedback = "\n".join(str(message.get("content")) for message in tool_messages)
            expected = ("AssertionError", "[exit code: 1]") if self.server.no_repair else ("public checks passed",)
            if not tool_messages or not all(marker in feedback for marker in expected):
                raise RuntimeError("DS did not receive the real public test output")
            if self.server.checked_patch:
                verified = tuple(f"VERIFIED {name}: sha256 " for name in (
                    "src/normalization.py", "src/helpers.py", "src/legacy.py", "src/solution.py",
                ))
                expected_patch = (*verified, "PATCH VERIFIED: 4 file(s) changed", "@@", "public checks passed")
                if not all(marker in feedback for marker in expected_patch):
                    raise RuntimeError("DS did not receive checked patch hashes, diff, and public test output")
                self.server.checked_patch_evidence = {
                    "verified_paths": list(verified), "diff_seen": "@@" in feedback,
                    "patch_success_seen": "PATCH VERIFIED: 4 file(s) changed" in feedback,
                    "public_test_seen": "public checks passed" in feedback,
                }
            raw = ("The public test failed; no repair was made." if self.server.no_repair
                   else "I edited the source and ran the public test.")
        action_ids = [101 + ordinal * 2, 102 + ordinal * 2]
        response = {"id": f"synthetic-ds-{ordinal}", "model": "transport-test", "created": 1,
                    "prompt_token_ids": [10, 20 + ordinal],
                    "choices": [{"index": 0, "token_ids": action_ids,
                                 "finish_reason": "tool_calls" if calls else "stop",
                                 "message": {"role": "assistant", "content": None if calls else raw,
                                             "tool_calls": calls},
                                 "logprobs": {"content": [{"token_id": token, "token": "fixture", "logprob": -0.2}
                                                          for token in action_ids]}}],
                    "hyper_tool_protocol": [{"parser_input": raw, "engine_text": raw, "decoded_tokens": raw,
                                             "token_ids": action_ids, "parser_result": {"tool_calls": calls}}]}
        self.server.responses.append(response)
        encoded = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args: object) -> None:
        """Suppress HTTP access logs to keep experiment evidence focused."""
        pass


def _checked_patch_block(reference: dict[str, Any]) -> str:
    """Build one complete patch from the controlled reference fixture."""
    normalization = reference["src/normalization.py"]
    if not isinstance(normalization, str) or not normalization.endswith("\n"):
        raise ValueError("word_counts normalization reference must be a complete source file")
    addition = "\n".join("+" + line for line in normalization.splitlines())
    return ("*** Begin Patch\n"
            "*** Add File: src/normalization.py\n"
            f"{addition}\n"
            "*** Update File: src/helpers.py\n"
            "@@\n"
            "-def words(text: str) -> list[str]:\n"
            "-    \"\"\"Return words extracted from one input string.\"\"\"\n"
            "-    return text.split()\n"
            "+from src.normalization import words\n"
            "*** Delete File: src/legacy.py\n"
            "*** Update File: src/solution.py\n"
            "@@\n"
            "-            result[word] = 1\n"
            "+            result[word] = result.get(word, 0) + 1\n"
            "*** End Patch")


def _checked_patch_command(reference: dict[str, Any]) -> str:
    """Feed the patch to the candidate helper and test only after a verified edit."""
    return ("python /opt/hyper-codex-home/checked_patch.py <<'PATCH' && python public_test.py\n"
            + _checked_patch_block(reference) + "\nPATCH")


async def run(args: argparse.Namespace) -> None:
    """Run a scripted repository episode and verify its frozen artifact and grade."""
    if args.checked_patch and (args.no_repair or args.budget_stop):
        raise ValueError("--checked-patch is a two-call repaired episode")
    if args.compaction_check and (args.no_repair or args.budget_stop or args.checked_patch):
        raise ValueError("--compaction-check must run without other scripted modes")
    args.output.mkdir(parents=True, exist_ok=False)
    fixture, _ = load_fixture("word_counts")
    patch = json.dumps(fixture["reference"])
    edit = ("import json,pathlib; p=pathlib.Path('/workspace'); "
            "print((p/'src/solution.py').read_text()); "
            f"changes=json.loads({patch!r}); "
            "[(p/name).unlink() if text is None else (p/name).write_text(text) for name,text in changes.items()]")
    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    backend.requests, backend.responses = [], []
    backend.no_repair = args.no_repair
    backend.checked_patch = args.checked_patch
    backend.checked_patch_evidence = {}
    backend.compaction_check = args.compaction_check
    backend.compaction_calls = []
    backend.normal_calls = 0
    backend.command = "python -c " + shlex.quote(edit) + " && python public_test.py"
    if args.compaction_check:
        backend.command = ("python -c \"import random,string; r=random.Random(42); "
                           "print(''.join(r.choices(string.ascii_letters,k=3600)))\"")
    elif args.checked_patch:
        backend.command = _checked_patch_command(fixture["reference"])
    elif args.no_repair:
        backend.command = "cat src/solution.py && python public_test.py"
    worker = threading.Thread(target=backend.serve_forever, daemon=True)
    worker.start()
    gateway = DeepSeekGateway("127.0.0.1", 0, f"http://127.0.0.1:{backend.server_port}",
                              "transport-test", 30)
    gateway.start()
    admin_url = f"http://127.0.0.1:{gateway.address[1]}"
    config = {"task_factory": "examples.code_agent.task:build_task", "task_config": {},
              "workspace": {"image": args.image, "run_id": args.run_id, "max_concurrent": 1},
              "session_root": str(args.output / "sessions"), "version": "0.1.1rc1",
              "provider": "deepseek-official", "model": "transport-test", "reasoning_effort": "off",
              "max_turns": 1 if args.budget_stop else 24 if args.compaction_check else 3,
              "max_new_tokens": 1024 if args.compaction_check else 256,
              "max_episode_tokens": 16384 if args.compaction_check else 128,
              "temperature": 1., "top_p": 1., "top_k": -1,
              "timeout_seconds": 90, "request_timeout": 30}
    prompt = adapt_row({"task_id": "repository-v1:word_counts", "ground_truth": task_manifest("word_counts"),
                        "prompt": [{"role": "user",
                                    "content": "Read and repair the repository; run public tests."}]}, 0)
    prompt = replace(prompt, metadata={**prompt.metadata, "input_ids": torch.tensor([10, 20])})
    try:
        program = DeepSeekAgentProgram(prompt, 7, 0, admin_url + "/v1", admin_url, config, None,
                                       admin_token=gateway.admin_token)
        rows = await program.run()
        first_request = backend.requests[0]
        visible_tools = [tool["function"]["name"] for tool in first_request.get("tools", [])]
        bash_schema = next(tool["function"]["parameters"] for tool in first_request.get("tools", [])
                           if tool["function"]["name"] == "bash")
        if bash_schema.get("additionalProperties") is not False:
            raise RuntimeError("Repository model did not see a closed bash argument schema")
        feedback_messages = ([] if len(backend.requests) < 2 else
                             [message for message in backend.requests[1]["messages"]
                              if message.get("role") == "tool"])
        result = {"reward": rows[0].reward, "calls": len(rows),
                  "tool_names": [call["function"]["name"] for call in backend.responses[0]["choices"][0]
                                 ["message"]["tool_calls"]],
                  "visible_tools": visible_tools,
                  "bash_schema_closed": bash_schema["additionalProperties"] is False,
                  "first_tool_arguments": json.loads(backend.responses[0]["choices"][0]
                                                     ["message"]["tool_calls"][0]["function"]["arguments"]),
                  "tool_feedback": [str(message.get("content"))[-1200:] for message in feedback_messages],
                  "checked_patch_evidence": backend.checked_patch_evidence,
                  "compaction_calls": backend.compaction_calls,
                  "normal_calls": backend.normal_calls,
                  "prompt_message_counts": [len(request.get("messages", [])) for request in backend.requests],
                  "truncated": rows[0].truncated, "terminal_reason": rows[0].terminal_reason,
                  "metadata": {key: value for key, value in rows[0].metadata.items()
                               if key in {"status", "reason", "artifact_hash", "patch_sha256",
                                          "repository_budget_truncated", "budget_termination"}},
                  "requests": len(backend.requests)}
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        expected = 0. if args.no_repair or args.compaction_check else 1.
        expected_calls = 1 if args.budget_stop else len(backend.requests) if args.compaction_check else 2
        if (result["reward"] != expected or result["requests"] != expected_calls
                or result["calls"] != expected_calls):
            raise RuntimeError(f"DS candidate/grader result violated the scripted contract: {result}")
        if args.budget_stop and (not result["truncated"] or result["terminal_reason"] != "max_completions"
                                 or result["metadata"].get("repository_budget_truncated") is not True
                                 or result["metadata"].get("budget_termination", {}).get("completed") != 1):
            raise RuntimeError(f"DS budget termination did not preserve its sampled call: {result}")
        if args.budget_stop:
            terminal_events = [event for event in rows[0].metadata.get("deepseek_events", [])
                               if event.get("type") == "turn/end"]
            reason = terminal_events[-1].get("data", {}).get("reason", {}) if terminal_events else {}
            error = reason.get("error", {})
            if (reason.get("kind") != "error" or error.get("status") != 409
                    or error.get("code") != "HTTP_409"
                    or error.get("message") != "DeepSeek repository reached max_completions=1"):
                raise RuntimeError(f"DS SDK budget terminal event is not the exact Gateway 409: {reason}")
            result["sdk_budget_error"] = error
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if args.checked_patch and (not result["checked_patch_evidence"].get("patch_success_seen")
                                   or not result["checked_patch_evidence"].get("diff_seen")
                                   or not result["checked_patch_evidence"].get("public_test_seen")
                                   or len(result["checked_patch_evidence"].get("verified_paths", [])) != 4
                                   or result["metadata"].get("status") != "passed"):
            raise RuntimeError(f"DS checked patch did not reach model and grader: {result}")
        if args.compaction_check:
            if (not backend.compaction_calls or backend.normal_calls != 18
                    or result["reward"] != 0.0 or len(rows) != len(backend.requests)):
                raise RuntimeError(f"DS compaction did not preserve the scripted episode: {result}")
            compact_ordinal = backend.compaction_calls[0]
            if (compact_ordinal + 1 >= len(backend.requests)
                    or len(backend.requests[compact_ordinal + 1].get("messages", []))
                    >= len(backend.requests[compact_ordinal - 1].get("messages", []))):
                raise RuntimeError(f"DS compaction did not shorten the following model prompt: {result}")
        print(json.dumps(result))
    finally:
        await asyncio.to_thread(gateway.close)
        backend.shutdown()
        worker.join(timeout=3)
        backend.server_close()


def main() -> None:
    """Parse experiment controls and run the repository probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--budget-stop", action="store_true")
    parser.add_argument("--checked-patch", action="store_true")
    parser.add_argument("--compaction-check", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
