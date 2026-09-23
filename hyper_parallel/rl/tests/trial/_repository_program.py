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
"""Real Docker/Codex/Gateway wiring with explicitly synthetic model responses, no NPU."""

import argparse
import asyncio
from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import subprocess
import threading

import torch

from rl.agentic.codex.gateway import CodexGateway, REPOSITORY_COMPACT_PROMPT
from rl.agentic.codex.harness import CodexAgentProgram
from rl.dataset.episodes import episode_rows

from examples.code_agent.prepare_data import adapt_row
from examples.code_agent.task import load_fixture, task_manifest


_LONG_OUTPUT_HEAD = "LONG_OUTPUT_HEAD_MARKER\n"
_LONG_OUTPUT_MIDDLE = "\nLONG_OUTPUT_MIDDLE_MARKER\n"
_LONG_OUTPUT_TAIL = "\nLONG_OUTPUT_TAIL_MARKER\n"
_LONG_OUTPUT_PATH = "/tmp/code-agent-long-output.txt"
_PROBE_SOURCE = '"""Checked patch transport probe."""'
_PROBE_PATH = "src/agent_probe.py"
_COMPACT_SUMMARY = "COMPACT_STATE_MARKER: long file read succeeded; no repository file was edited."


def require(condition: bool, message: str) -> None:
    """Raise a functional acceptance error without implying real model learning."""
    if not condition:
        raise RuntimeError(message)


class Backend(BaseHTTPRequestHandler):
    """Serve a scripted tool call followed by a final response for transport acceptance."""

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Return synthetic token evidence for a controlled read/edit/public-test command."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        ordinal = len(self.server.requests) - 1
        calls = []
        if ordinal == 0 or (self.server.stdin_schema_check and ordinal == 1):
            tools = [item["function"] for item in body.get("tools", [])]
            selected = next((tool for tool in tools if "exec_command" in tool["name"]), None)
            if selected is None and not self.server.checked_patch_check:
                selected = next((tool for tool in tools if "shell" in tool["name"]), None)
            if selected is None:
                self.send_error(500, "Pinned CLI did not declare a shell tool")
                return
            properties = selected.get("parameters", {}).get("properties", {})
            arguments = {"cmd": self.server.command} if "cmd" in properties else {"command": self.server.command}
            if properties.get("command", {}).get("type") == "array":
                arguments["command"] = ["/bin/bash", "-lc", self.server.command]
            if self.server.stdin_schema_check:
                require(selected["name"] == "exec_command" and "stdin" not in properties,
                        "Pinned CLI did not present the expected closed exec_command schema")
                if ordinal == 0:
                    arguments = {"cmd": "python /opt/hyper-codex-home/checked_patch.py",
                                 "stdin": _checked_patch_block()}
                else:
                    visible = "\n".join(str(message.get("content", "")) for message in body["messages"])
                    tool_messages = [message for message in body["messages"] if message.get("role") == "tool"]
                    require(not tool_messages and "does not accept argument(s) stdin" in visible
                            and "cmd shell command using a heredoc" in visible
                            and "<<'PATCH'\n*** Begin Patch" in visible,
                            "Schema correction did not reach the next model request before CLI execution")
                    self.server.stdin_schema_evidence = {"precise_feedback_seen": True,
                                                         "no_cli_tool_feedback_before_resample": True}
            calls = [{"id": "scripted-tool", "type": "function", "function": {
                "name": selected["name"], "arguments": json.dumps(arguments),
            }}]
            raw = "<tool_call>" + json.dumps({"name": selected["name"], "arguments": arguments}) + "</tool_call>"
        elif self.server.compaction_check and ordinal == 1:
            require(body.get("tools") in (None, []) and body["messages"][-1] == {
                "role": "user", "content": REPOSITORY_COMPACT_PROMPT,
            }, "Compaction backend request retained tools or changed the pinned summary prompt")
            context = "\n".join(str(message.get("content", "")) for message in body["messages"])
            require("network=none" in context and 'type="candidate-container"' in context
                    and "Network access is enabled" not in context,
                    "Compaction model context received stale container permissions")
            self.server.compaction_evidence = {"backend_no_tools": True, "pinned_prompt_seen": True,
                                               "candidate_permissions_seen": True}
            raw = _COMPACT_SUMMARY
        else:
            tool_outputs = [message.get("content", "") for message in body["messages"] if message["role"] == "tool"]
            if self.server.compaction_check:
                visible = {item.get("function", {}).get("name") for item in body.get("tools", [])}
                require(ordinal == 2 and visible == {"exec_command", "write_stdin"}
                        and any(_COMPACT_SUMMARY in str(message.get("content", ""))
                                for message in body["messages"]),
                        "Normal model turn after compaction lost its tools or state summary")
                raw = "The long file was read; no repository edit was made."
            elif self.server.checked_patch_check or self.server.stdin_schema_check:
                outputs = [value for value in tool_outputs if isinstance(value, str)]
                valid = [value for value in outputs
                         if ("Process exited with code 0" in value
                             and f"VERIFIED {_PROBE_PATH}: sha256 absent -> " in value
                             and f"+++ after/{_PROBE_PATH}" in value
                             and "+" + _PROBE_SOURCE in value
                             and "PATCH VERIFIED: 1 file(s) changed" in value)]
                self.server.checked_patch_evidence = {"tool_response_count": len(outputs),
                                                      "verified_feedback_count": len(valid),
                                                      "feedback_sha256": hashlib.sha256(valid[0].encode()).hexdigest()
                                                      if valid else None}
                require(len(valid) == 1, "Model did not receive the checked patch's real exit, hash and diff")
                raw = "The checked patch changed one source file; the repository grader will score it."
            elif self.server.long_output_check:
                outputs = [value for value in tool_outputs if isinstance(value, str)]
                expected = self.server.long_output
                self.server.long_output_evidence = {
                    "tool_response_count": len(outputs),
                    "tool_response_chars": [len(value) for value in outputs],
                    "head_seen": any(_LONG_OUTPUT_HEAD in value for value in outputs),
                    "middle_seen": any(_LONG_OUTPUT_MIDDLE in value for value in outputs),
                    "tail_seen": any(_LONG_OUTPUT_TAIL in value for value in outputs),
                    "full_output_seen": any(expected in value for value in outputs),
                    "truncation_marker_seen": any("truncated output" in value or "chars truncated" in value
                                                  for value in outputs),
                }
                raw = "The long file was read; the repository was not changed."
            else:
                require(any("public checks passed" in str(value) for value in tool_outputs),
                        "CLI did not run public tests")
                raw = "Repository changes and public checks completed."
        action_ids = [101 + ordinal * 2, 102 + ordinal * 2]
        response = {"id": f"synthetic-{ordinal}", "model": "transport-test", "created": 1,
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
        """Keep candidate/tool contents out of routine server logs."""


def _long_output() -> str:
    """Place a marker beyond either half of the observed 10,000-character CLI window."""
    prefix = _LONG_OUTPUT_HEAD + "A" * (15000 - len(_LONG_OUTPUT_HEAD))
    return (prefix + _LONG_OUTPUT_MIDDLE
            + "B" * (30010 - len(prefix) - len(_LONG_OUTPUT_MIDDLE) - len(_LONG_OUTPUT_TAIL))
            + _LONG_OUTPUT_TAIL)


def _long_output_command() -> str:
    """Create a deterministic file outside the repository and read it with the real shell tool."""
    script = ("from pathlib import Path; "
              f"head={_LONG_OUTPUT_HEAD!r}; middle={_LONG_OUTPUT_MIDDLE!r}; tail={_LONG_OUTPUT_TAIL!r}; "
              "prefix=head+'A'*(15000-len(head)); "
              "text=prefix+middle+'B'*(30010-len(prefix)-len(middle)-len(tail))+tail; "
              f"Path({_LONG_OUTPUT_PATH!r}).write_text(text)")
    return "python -c " + shlex.quote(script) + " && cat " + shlex.quote(_LONG_OUTPUT_PATH)


def _checked_patch_command() -> str:
    """Exercise the candidate-local helper without placing patch files in the submission."""
    return "python /opt/hyper-codex-home/checked_patch.py <<'PATCH'\n" + _checked_patch_block() + "\nPATCH"


def _checked_patch_block() -> str:
    """Supply one deterministic source addition to the helper's stdin."""
    return ("*** Begin Patch\n"
            f"*** Add File: {_PROBE_PATH}\n"
            f"+{_PROBE_SOURCE}\n"
            "*** End Patch")


async def run(args: argparse.Namespace) -> None:
    """Drive actual CLI tools, freeze/judge the submission and check exact supplied token mapping."""
    args.output.mkdir(parents=True, exist_ok=True)
    fixture, _ = load_fixture("word_counts")
    patch = json.dumps(fixture["reference"])
    edit = ("import json,pathlib; p=pathlib.Path('/workspace'); "
            "print((p/'src/solution.py').read_text()); "
            f"changes=json.loads({patch!r}); "
            "[(p/name).unlink() if text is None else (p/name).write_text(text) for name,text in changes.items()]")
    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    backend.requests, backend.responses = [], []
    backend.long_output_check = args.long_output_check
    backend.long_output = _long_output() if args.long_output_check or args.compaction_check else ""
    backend.long_output_evidence = {}
    backend.checked_patch_check = args.checked_patch_check
    backend.checked_patch_evidence = {}
    backend.stdin_schema_check = args.stdin_schema_check
    backend.stdin_schema_evidence = {}
    backend.compaction_check = args.compaction_check
    backend.compaction_evidence = {}
    backend.command = "python -c " + shlex.quote(edit) + " && python public_test.py"
    if args.checked_patch_check or args.stdin_schema_check:
        backend.command = _checked_patch_command()
    elif args.long_output_check or args.compaction_check:
        backend.command = _long_output_command()
    elif args.no_repair:
        backend.command = "cat src/solution.py && python public_test.py"
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    gateway = CodexGateway("127.0.0.1", 0, f"http://127.0.0.1:{backend.server_port}", "transport-test", 10)
    gateway.start()
    url = f"http://127.0.0.1:{gateway.address[1]}"
    config = {"task_factory": "examples.code_agent.task:build_task", "task_config": {},
              "workspace": {"image": args.image, "run_id": args.run_id, "max_concurrent": 1},
              "session_root": str(args.output / "sessions"), "executable": "codex", "version": "0.152.1",
              "model": "transport-test", "model_context_window": 40960,
              "max_turns": 1 if args.budget_stop else 3,
              "max_new_tokens": 256, "max_episode_tokens": 128,
              "temperature": 1., "top_p": 1., "top_k": -1, "timeout_seconds": 90, "request_timeout": 10}
    if args.compaction_check:
        config["model_context_window"] = 8192
    prompt = adapt_row({"task_id": "repository-v1:word_counts", "ground_truth": task_manifest("word_counts"),
                        "prompt": [{"role": "user", "content": "Read and repair the repository; run public tests."}]},
                       0)
    prompt = replace(prompt, metadata={**prompt.metadata, "input_ids": torch.tensor([10, 20])})
    try:
        program = CodexAgentProgram(prompt, 7, 0, url, config, None,
                                    admin_url=url, admin_token=gateway.admin_token)
        rows = await program.run()
        expected_calls = 1 if args.budget_stop else 3 if (args.compaction_check or args.stdin_schema_check) else 2
        expected_reward = 0. if any((args.no_repair, args.long_output_check, args.checked_patch_check,
                                     args.compaction_check, args.stdin_schema_check)) else 1.
        require(len(rows) == len(backend.responses) == expected_calls, "Unexpected number of sampled model calls")
        first = backend.requests[0]
        visible_tools = [item.get("function", {}) for item in first.get("tools", [])]
        require({item.get("name") for item in visible_tools} == {"exec_command", "write_stdin"}
                and len(visible_tools) == 2, "Repository model saw tools outside the executable CLI pair")
        model_context = "\n".join(str(item.get("content", "")) for item in first.get("messages", []))
        require("network=none" in model_context and 'type="candidate-container"' in model_context
                and "Network access is enabled" not in model_context
                and "<skills_instructions>" not in model_context,
                "Repository model received stale permissions or skills")
        exec_tool = next(item for item in visible_tools if item["name"] == "exec_command")
        exec_properties = exec_tool.get("parameters", {}).get("properties", {})
        require(all(name not in exec_properties for name in ("sandbox_permissions", "justification", "prefix_rule")),
                "Repository model saw unavailable approval arguments")
        for row, response in zip(rows, backend.responses):
            choice = response["choices"][0]
            require(row.token_ids.tolist() == response["prompt_token_ids"] + choice["token_ids"], "Token IDs changed")
            require(row.action_mask.tolist() == [False, False, True, True], "Action mask changed")
            require(torch.allclose(row.rollout_log_probs, torch.tensor([0., -.2, -.2])), "Logprob mapping changed")
            require(row.reward == expected_reward and row.policy_version == 7,
                    "Episode reward or policy identity differs")
            if args.budget_stop:
                require(row.truncated and row.terminal_reason == "max_completions",
                        "Canonical trajectory lost its budget truncation status")
                require(row.metadata.get("repository_budget_truncated") is True,
                        "Budget termination was not recorded as truncation")
                require(row.metadata.get("budget_termination") == {
                    "reason": "max_completions", "limit": 1, "completed": 1, "policy_version": 7,
                }, "Budget termination evidence changed")
                require(choice["finish_reason"] == "tool_calls", "Budget stop invented a final model response")
        groups = episode_rows(rows)
        require(len(groups) == 1, "Calls were treated as separate reward episodes")
        artifacts = list((args.output / "sessions").glob("*/submission.manifest.json"))
        require(len(artifacts) == 1, "Submission grading evidence missing")
        events = list((args.output / "sessions").glob("*/codex-events.jsonl"))
        require(len(events) == 1 and "command_execution" in events[0].read_text(), "No real CLI tool event")
        require(not gateway._server.state.sessions, "Program left an active Gateway session")
        if args.compaction_check:
            manifest = json.loads(artifacts[0].read_text())
            require(manifest["changes"] == {"added": [], "modified": [], "deleted": []}
                    and manifest["content_hash"] == rows[0].metadata["base_hash"],
                    "Compaction read unexpectedly changed the frozen repository")
            require(rows[0].metadata.get("evaluated") == len(fixture["cases"]) and rows[0].reward == 0.,
                    "Compaction baseline did not reach the independent grader with reward zero")
            completed = [item["item"] for item in map(json.loads, events[0].read_text().splitlines())
                         if item.get("type") == "item.completed"
                         and item.get("item", {}).get("type") == "command_execution"]
            actual = [item for item in completed if _LONG_OUTPUT_PATH in item.get("command", "")]
            require(len(actual) == 1 and actual[0].get("exit_code") == 0
                    and actual[0].get("aggregated_output") == backend.long_output,
                    "Compaction probe did not read the complete 30,010-character file")
            gateway_events = list((args.output / "sessions").glob("*/gateway-events.jsonl"))
            require(len(gateway_events) == 1, "Compaction Gateway trace missing")
            records = [item["payload"] for item in map(json.loads, gateway_events[0].read_text().splitlines())
                       if item.get("type") == "completion.recorded"]
            require(len(records) == 3, "Pinned CLI did not make tool, compact and normal calls in order")
            kinds = [json.loads(record["original_request"]["client_metadata"]["x-codex-turn-metadata"])
                     ["request_kind"] for record in records]
            require(kinds == ["turn", "compaction", "turn"], "Pinned CLI compaction metadata or order differs")
            compact_original = records[1]["original_request"]
            require(compact_original.get("tools") == []
                    and compact_original["input"][-1] == {"type": "message", "role": "user",
                                                          "content": [{"type": "input_text",
                                                                       "text": REPOSITORY_COMPACT_PROMPT}]}
                    and records[1]["request"].get("tools") in (None, [])
                    and {item["function"]["name"] for item in records[2]["request"]["tools"]}
                    == {"exec_command", "write_stdin"},
                    "Compaction or following normal turn changed the pinned tools/prompt contract")
            require(backend.compaction_evidence.get("candidate_permissions_seen")
                    and backend.compaction_evidence.get("backend_no_tools")
                    and backend.compaction_evidence.get("pinned_prompt_seen"),
                    "Backend did not verify the model-side compact request")
            backend.compaction_evidence.update({"request_kinds": kinds,
                                                "context_window": config["model_context_window"],
                                                "cli_output_chars": len(actual[0]["aggregated_output"]),
                                                "baseline_unchanged": True,
                                                "grader_cases": rows[0].metadata["evaluated"],
                                                "reward": rows[0].reward})
        if args.checked_patch_check or args.stdin_schema_check:
            manifest = json.loads(artifacts[0].read_text())
            require(manifest["changes"] == {"added": [_PROBE_PATH], "modified": [], "deleted": []}
                    and manifest["content_hash"] != rows[0].metadata["base_hash"]
                    and manifest["content_hash"] == rows[0].metadata["artifact_hash"],
                    "Checked patch did not survive as exactly one added source file")
            require(rows[0].metadata.get("evaluated") == len(fixture["cases"]) and rows[0].reward == 0.,
                    "Checked patch was not independently graded as the still-broken repository")
            completed = [item["item"] for item in map(json.loads, events[0].read_text().splitlines())
                         if item.get("type") == "item.completed"
                         and item.get("item", {}).get("type") == "command_execution"]
            actual = [item for item in completed if "checked_patch.py" in item.get("command", "")]
            require(len(actual) == 1 and actual[0].get("exit_code") == 0
                    and f"VERIFIED {_PROBE_PATH}" in actual[0].get("aggregated_output", ""),
                    "Pinned CLI did not execute the checked patch successfully")
            gateway_events = list((args.output / "sessions").glob("*/gateway-events.jsonl"))
            require(len(gateway_events) == 1, "Checked patch Gateway trace missing")
            records = [item["payload"] for item in map(json.loads, gateway_events[0].read_text().splitlines())
                       if item.get("type") == "completion.recorded"]
            require(len(records) == expected_calls, "Checked patch Gateway trace has an unexpected call count")
            raw_outputs = [item.get("output", "") for item in records[-1]["original_request"].get("input", [])
                           if isinstance(item, dict) and item.get("type") == "function_call_output"]
            require(any(actual[0]["aggregated_output"] in value for value in raw_outputs)
                    and backend.checked_patch_evidence.get("verified_feedback_count") == 1,
                    "Checked patch feedback was not present in the next raw model request")
            backend.checked_patch_evidence.update({"artifact_hash": manifest["content_hash"],
                                                   "grader_cases": rows[0].metadata["evaluated"],
                                                   "reward": rows[0].reward,
                                                   "original_request_feedback_seen": True})
            if args.stdin_schema_check:
                rejected = records[0]
                require(rejected["metadata"].get("failure_origin") == "model"
                        and rejected["metadata"].get("failure_reason") == "undeclared_tool_argument"
                        and rejected["metadata"].get("trainable") is True
                        and rejected["metadata"].get("invalid_arguments") == ["stdin"],
                        "Undeclared stdin was not kept as a verified malformed model action")
                require(rejected["response"] == backend.responses[0]
                        and rejected["response"]["choices"][0]["token_ids"] == [101, 102]
                        and rejected["response"]["prompt_token_ids"] == [10, 20],
                        "Rejected tool call lost its original sampled response or token identity")
                require(records[1]["metadata"].get("failure_origin") is None
                        and backend.stdin_schema_evidence.get("precise_feedback_seen"),
                        "Valid heredoc retry was not sampled after precise schema feedback")
                backend.stdin_schema_evidence.update({"invalid_call_never_executed": len(actual) == 1,
                                                      "valid_heredoc_edited": True,
                                                      "artifact_hash": manifest["content_hash"],
                                                      "grader_cases": rows[0].metadata["evaluated"],
                                                      "reward": rows[0].reward})
        if args.long_output_check:
            manifest = json.loads(artifacts[0].read_text())
            require(rows[0].metadata["base_hash"] == manifest["content_hash"]
                    and manifest["changes"] == {"added": [], "modified": [], "deleted": []},
                    "Long-output read changed the repository baseline")
            require(rows[0].metadata.get("evaluated", 0) > 0,
                    "Long-output baseline did not reach the independent repository grader")
            completed = [item["item"] for item in map(json.loads, events[0].read_text().splitlines())
                         if item.get("type") == "item.completed"
                         and item.get("item", {}).get("type") == "command_execution"]
            actual = [item for item in completed if _LONG_OUTPUT_PATH in item.get("command", "")]
            require(len(actual) == 1 and actual[0].get("exit_code") == 0
                    and actual[0].get("aggregated_output") == backend.long_output,
                    "Pinned CLI did not execute the complete 30,010-character cat")
            gateway_events = list((args.output / "sessions").glob("*/gateway-events.jsonl"))
            require(len(gateway_events) == 1, "Long-output Gateway trace missing")
            records = [item["payload"] for item in map(json.loads, gateway_events[0].read_text().splitlines())
                       if item.get("type") == "completion.recorded"]
            require(len(records) == 2, "Long-output Gateway trace has an unexpected call count")
            raw_outputs = [item.get("output", "") for item in records[1]["original_request"].get("input", [])
                           if isinstance(item, dict) and item.get("type") == "function_call_output"]
            require(any(backend.long_output in value for value in raw_outputs),
                    "Pinned CLI omitted the complete file from its next raw Responses request")
            backend.long_output_evidence.update({
                "cli_output_chars": len(actual[0]["aggregated_output"]),
                "cli_output_sha256": hashlib.sha256(backend.long_output.encode()).hexdigest(),
                "original_request_full_output_seen": True,
                "baseline_unchanged": True,
                "grader_cases": rows[0].metadata["evaluated"],
                "reward": rows[0].reward,
            })
            evidence = backend.long_output_evidence
            require(evidence.get("head_seen") and evidence.get("middle_seen") and evidence.get("tail_seen")
                    and evidence.get("full_output_seen") and not evidence.get("truncation_marker_seen"),
                    "Pinned CLI truncated long tool feedback; see long-output-evidence.json")
        summary = {"status": "passed", "model_backend": "scripted_test_responses_not_a_model",
                   "real_components": ["Docker", "Codex 0.152.1", "Gateway", "stdio relay", "repository grader"],
                   "calls": len(rows), "episodes": len(groups), "reward": rows[0].reward,
                   "budget_stop": args.budget_stop,
                   "reference_repair": not any((args.no_repair, args.long_output_check,
                                                args.checked_patch_check, args.compaction_check,
                                                args.stdin_schema_check)),
                   "policy_version": 7, "npu_count": 0, "image": args.image,
                   "artifact_hash": rows[0].metadata["artifact_hash"]}
        if args.long_output_check:
            summary["long_output_evidence"] = backend.long_output_evidence
        if args.checked_patch_check:
            summary["checked_patch_evidence"] = backend.checked_patch_evidence
        if args.stdin_schema_check:
            summary["stdin_schema_evidence"] = backend.stdin_schema_evidence
        if args.compaction_check:
            summary["compaction_evidence"] = backend.compaction_evidence
    finally:
        (args.output / "scripted-requests.json").write_text(json.dumps(backend.requests, indent=2) + "\n")
        if args.long_output_check:
            (args.output / "long-output-evidence.json").write_text(
                json.dumps(backend.long_output_evidence, indent=2) + "\n")
        if args.checked_patch_check:
            (args.output / "checked-patch-evidence.json").write_text(
                json.dumps(backend.checked_patch_evidence, indent=2) + "\n")
        if args.stdin_schema_check:
            (args.output / "stdin-schema-evidence.json").write_text(
                json.dumps(backend.stdin_schema_evidence, indent=2) + "\n")
        if args.compaction_check:
            (args.output / "compaction-evidence.json").write_text(
                json.dumps(backend.compaction_evidence, indent=2) + "\n")
        await asyncio.to_thread(gateway.close)
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=5)
        remaining = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=hyper-rl.run-id={args.run_id}"],
                                   capture_output=True, text=True, check=True, timeout=30)
        require(not remaining.stdout.strip(), "Experiment left owned containers")
    summary["remaining_owned_containers"] = []
    (args.output / "completed.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def main() -> None:
    """Select the already validated offline candidate image and an unused evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--budget-stop", action="store_true", help="Stop after the real tool call by exhausting quota")
    parser.add_argument("--no-repair", action="store_true", help="Keep the broken baseline to verify a real zero score")
    parser.add_argument("--long-output-check", action="store_true",
                        help="Check whether the pinned CLI preserves a 30,010-character cat in the next model request")
    parser.add_argument("--checked-patch-check", action="store_true",
                        help="Check candidate-local patch feedback and nonempty frozen source in the real CLI")
    parser.add_argument("--stdin-schema-check", action="store_true",
                        help="Reject undeclared stdin, resample a valid heredoc, then freeze its source edit")
    parser.add_argument("--compaction-check", action="store_true",
                        help="Trigger and verify pinned CLI auto compaction after a large tool response")
    args = parser.parse_args()
    if args.no_repair and not args.budget_stop:
        parser.error("--no-repair requires --budget-stop")
    if args.long_output_check and (args.no_repair or args.budget_stop):
        parser.error("--long-output-check cannot be combined with --no-repair or --budget-stop")
    if args.checked_patch_check and (args.no_repair or args.budget_stop or args.long_output_check
                                     or args.stdin_schema_check):
        parser.error("--checked-patch-check must run without other special modes")
    if args.stdin_schema_check and (args.no_repair or args.budget_stop or args.long_output_check
                                    or args.compaction_check):
        parser.error("--stdin-schema-check must run without other special modes")
    if args.compaction_check and (args.no_repair or args.budget_stop or args.long_output_check
                                  or args.checked_patch_check or args.stdin_schema_check):
        parser.error("--compaction-check must run without other special modes")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
