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
"""Probe genuine repository repair against an already running native vLLM service."""

import argparse
import asyncio
from dataclasses import replace
import importlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
import yaml

from rl.agentic.codex.gateway import CodexGateway
from rl.agentic.codex.harness import CodexAgentProgram
from rl.agentic.ds_harness.gateway import DeepSeekGateway
from rl.agentic.ds_harness.harness import DeepSeekAgentProgram
from rl.dataset.episodes import episode_rows
from hyper_parallel.rl.tests.st._agent_train import _verify_captured_tokens


def _events(directory: Path, runner: str) -> dict:
    commands, failures = [], []
    if runner == "deepseek":
        path = directory / "deepseek-events.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        return {"deepseek_events": events}
    path = directory / "codex-events.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            item = event.get("item", {})
            if event.get("type") == "item.completed" and item.get("type") == "command_execution":
                commands.append({key: item.get(key) for key in ("command", "exit_code", "status")})
            if event.get("type") in {"error", "turn.failed"} or item.get("type") == "error":
                failures.append(event)
    return {"commands": commands, "cli_diagnostics": failures}


async def run(args: argparse.Namespace) -> None:
    """Retain all sampled outcomes; transport or unknown failures abort the probe."""
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    agentic, rollout = config["agentic"], config["rollout"]
    runner = agentic["runner"]
    if runner not in {"codex", "deepseek"}:
        raise ValueError("Repository probe requires codex or deepseek runner")
    settings = {**agentic[runner], "max_turns": agentic["max_turns"],
                "max_episode_tokens": agentic.get("max_episode_tokens"),
                **{key: rollout.get(key) for key in
                   ("max_new_tokens", "temperature", "top_p", "top_k", "seed", "ignore_eos")}}
    args.output.mkdir(parents=True, exist_ok=False)
    settings["session_root"] = str(args.output / "sessions")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer_path"], local_files_only=True)
    module, name = config["data"]["row_adapter"].split(":", 1)
    adapt_row = getattr(importlib.import_module(module), name)
    prompts = [adapt_row(row, index) for index, row in
               enumerate(pq.read_table(config["data"]["train_path"]).to_pylist())]
    if args.fixture:
        prompts = [prompt for prompt in prompts if prompt.ground_truth["fixture_id"] == args.fixture]
    if args.instance:
        prompts = [prompt for prompt in prompts if prompt.ground_truth.get("instance_id") == args.instance]
    if not prompts or len(prompts) * args.samples > 4:
        raise ValueError("Probe requires between one and four total candidates")
    gateway_class = CodexGateway if runner == "codex" else DeepSeekGateway
    gateway = gateway_class("127.0.0.1", 0, args.backend_url, settings["model"],
                           float(settings.get("request_timeout", 600)),
                           max_request_bytes=int(settings.get("max_request_bytes", 8 * 1024 * 1024)),
                           max_response_bytes=int(settings.get("max_response_bytes", 32 * 1024 * 1024)))
    summary = {"status": "running", "backend_url": args.backend_url, "policy_version": 0, "candidates": []}

    def save() -> None:
        """Persist completed candidates before starting another independent episode."""
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    gateway.start()
    url = f"http://127.0.0.1:{gateway.address[1]}"
    try:
        save()
        for prompt in prompts:
            encoded = tokenizer.apply_chat_template(
                [{"role": message.role, "content": message.content} for message in prompt.messages],
                tokenize=True, add_generation_prompt=True, return_dict=True,
            )
            prompt = replace(prompt, metadata={**prompt.metadata,
                                              "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long)})
            for sample in range(args.samples):
                record = {"task_id": prompt.prompt_id, "sample_index": sample}
                previous = set((args.output / "sessions").glob("*"))
                try:
                    if runner == "codex":
                        program = CodexAgentProgram(prompt, 0, sample, url, settings, tokenizer.eos_token_id,
                                                    admin_url=url, admin_token=gateway.admin_token)
                    else:
                        program = DeepSeekAgentProgram(prompt, 0, sample, url + "/v1", url, settings,
                                                       tokenizer.eos_token_id, admin_token=gateway.admin_token)
                    rows = await program.run()
                    if len(episode_rows(rows)) != 1:
                        raise RuntimeError("One repository candidate did not form exactly one complete episode")
                    for row in rows:
                        _verify_captured_tokens(row)
                    record.update(status="completed", reward=rows[0].reward, calls=len(rows),
                                  token_lengths=[{"prompt": row.token_ids.numel() - int(row.action_mask.sum()),
                                                  "action": int(row.action_mask.sum())} for row in rows],
                                  reward_metadata={key: rows[0].metadata[key] for key in
                                                   ("status", "reason", "failure_reason", "artifact_hash", "cases",
                                                    "report_path", "patch_sha256", "evaluated", "instance_id")
                                                   if key in rows[0].metadata})
                except Exception as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}")
                    summary["status"] = "failed"
                    raise
                finally:
                    created = [path for path in set((args.output / "sessions").glob("*")) - previous
                               if path.is_dir() and not path.name.startswith(".")]
                    record["artifacts"] = [{"directory": str(path), **_events(path, runner)} for path in sorted(created)]
                    summary["candidates"].append(record)
                    save()
        summary["status"] = "completed"
        save()
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        await asyncio.to_thread(gateway.close)


def main() -> None:
    """Run at most four real candidates without launching or changing the model service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--backend-url", default="http://127.0.0.1:18860")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--fixture", choices=("merge_intervals", "word_counts"))
    parser.add_argument("--instance", help="Select one SWE-bench instance from the configured dataset")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
