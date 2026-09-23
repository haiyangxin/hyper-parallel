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
"""Run existing isolated repository Code Agent tasks without training evidence."""

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path

import pyarrow.parquet as pq

from rl.agentic.codex.gateway import CodexGateway
from rl.agentic.codex.harness import CodexAgentProgram
from examples.code_agent.swebench_data import adapt_row as adapt_swebench_row
from examples.code_agent.prepare_data import adapt_row as adapt_fixture_row


async def run_inference(args: argparse.Namespace) -> dict:
    """Execute one prepared fixture or SWE-bench task against an external backend.

    Args:
        args: Controller arguments, including a JSON harness configuration and public parquet.

    Returns:
        A JSON-compatible scored inference result. No trajectory or sampled logprob is fabricated.
    """
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Inference configuration must be a JSON object")
    fixture_id = args.fixture_id
    identity_key = "fixture_id" if fixture_id else "instance_id"
    identity = fixture_id or args.instance_id
    expected_factory = ("examples.code_agent.task:build_task" if fixture_id
                        else "examples.code_agent.swebench_task:build_task")
    if config.get("task_factory") != expected_factory:
        raise ValueError("Task selector and existing task_factory must agree")
    config["model"] = args.model
    config["session_root"] = str(args.output.resolve() / "sessions")
    config.setdefault("max_turns", None)
    rows = [row for row in pq.read_table(args.data).to_pylist()
            if row.get("ground_truth", {}).get(identity_key) == identity]
    if len(rows) != 1:
        raise ValueError("Public data must contain exactly one requested instance")
    prompt = (adapt_fixture_row if fixture_id else adapt_swebench_row)(rows[0], 0)
    args.output.mkdir(parents=True, exist_ok=False)
    gateway = CodexGateway(
        "127.0.0.1", 0, args.backend_url, args.model, float(config.get("request_timeout", 1800)),
        inference_only=True, backend_key_file=args.backend_key_file,
        max_request_bytes=int(config.get("max_request_bytes", 8 * 1024 * 1024)),
        max_response_bytes=int(config.get("max_response_bytes", 32 * 1024 * 1024)),
    )
    manifest = {"inference_only": True, "trainable": False, "policy_version": None,
                "backend_url": args.backend_url, "model": args.model, "config": config,
                "prompt_id": prompt.prompt_id, "ground_truth": prompt.ground_truth}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    gateway.start()
    host, port = gateway.address
    url = f"http://{host}:{port}"
    try:
        program = CodexAgentProgram(prompt, None, 0, url, config, None,
                                    admin_url=url, admin_token=gateway.admin_token, inference_only=True)
        result = await program.run_inference()
        payload = asdict(result)
        payload["artifact_dir"] = str(result.artifact_dir)
        payload.update(inference_only=True, trainable=False, policy_version=None)
        (args.output / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps({
            "inference_only": True, "trainable": False, "status": "incomplete",
            "error_type": type(error).__name__, "error": str(error),
        }, indent=2), encoding="utf-8")
        raise
    finally:
        await asyncio.to_thread(gateway.close)


def main() -> None:
    """Parse controller-only paths and run one inference episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON containing Codex harness settings")
    parser.add_argument("--data", type=Path, required=True, help="Public parquet from prepare_data or swebench_data")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--instance-id")
    selector.add_argument("--fixture-id", choices=("merge_intervals", "word_counts"))
    parser.add_argument("--backend-url", required=True, help="Backend origin without /v1")
    parser.add_argument("--backend-key-file", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New controller-only output directory")
    asyncio.run(run_inference(parser.parse_args()))


if __name__ == "__main__":
    main()
