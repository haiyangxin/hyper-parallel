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
"""Verify real SWE-bench trajectories, official grading and functional training."""

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

import torch.distributed as dist
import yaml

from hyper_parallel.rl.tests.st._agent_train import AgentObservedTrainer, _validate_rank_alignment
from _repository_train import RepositoryObservedTrainer
from rl.dataset.episodes import episode_rows

from examples.code_agent.swebench_artifacts import _hash, _read


def _verify_syntax(item: dict, manifest: dict, hashes: dict) -> dict:
    """Bind compiler diagnostics to a healthy baseline and the frozen changed source."""
    evidence = json.loads(Path(item["syntax_path"]).read_text(encoding="utf-8"))
    baseline, candidate = evidence["baseline"], evidence["candidate"]
    changed = manifest["changes"]
    expected = {name: hashes[name] for name in changed["added"] + changed["modified"]}
    if (item["reward"] != 0 or item["failure_origin"] != "model" or item["trainable"] is not True
            or item["evaluated"] != 0 or evidence["artifact_hash"] != item["artifact_hash"]
            or evidence["patch_sha256"] != item["patch_sha256"] or baseline["errors"]
            or not candidate["errors"] or candidate["files"] != expected
            or set(baseline["files"]) != set(changed["modified"] + changed["deleted"])
            or candidate["python"] != baseline["python"] or not candidate["python"].startswith("3.9.")
            or candidate["executable"] != baseline["executable"]
            or candidate["executable"] != "/opt/miniconda3/envs/testbed/bin/python"):
        raise RuntimeError("SWE-bench syntax evidence differs from its frozen candidate or healthy baseline")
    for error in candidate["errors"]:
        if (error["path"] not in expected or error["type"] not in {"SyntaxError", "IndentationError", "TabError"}
                or not isinstance(error["lineno"], int) or error["lineno"] <= 0 or not error["message"]):
            raise RuntimeError("SWE-bench syntax diagnostic is incomplete")
    item.update(syntax_verified=True, changes=changed)
    return item


def verify_episode(metadata: dict, reward: float, call_count: int) -> dict:
    """Bind a real episode reward to frozen bytes and its official grader report."""
    item = {key: metadata.get(key) for key in (
        "episode_id", "instance_id", "status", "artifact", "artifact_hash", "patch_sha256",
        "report_path", "syntax_path", "evaluated", "budget_termination", "failure_origin", "trainable")}
    item.update(reward=reward, call_count=call_count, officially_graded=False)
    if reward not in (0., 1.):
        raise RuntimeError("SWE-bench reward must be binary")
    if not item["artifact"]:
        invalid = item["status"] == "invalid_submission" and bool(metadata.get("reason"))
        model_failure = metadata.get("failure_origin") == "model" and metadata.get("trainable") is True
        if reward != 0 or not (invalid or model_failure):
            raise RuntimeError("Ungraded episode lacks an explicit trainable zero-reward reason")
        return item
    manifest_path = Path(item["artifact"])
    archive = manifest_path.with_name("submission.tar")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files, modes = _read(archive, 16 * 1024 * 1024, 4096)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    if (hashes != manifest["files"] or modes != manifest["modes"]
            or _hash(files, modes) != manifest["content_hash"]
            or manifest["content_hash"] != item["artifact_hash"]):
        raise RuntimeError("SWE-bench frozen artifact differs from manifest or trajectory")
    patch = manifest_path.with_name("submission.patch").read_bytes()
    if (hashlib.sha256(patch).hexdigest() != manifest["patch_sha256"]
            or manifest["patch_sha256"] != item["patch_sha256"]):
        raise RuntimeError("SWE-bench patch SHA differs from manifest or trajectory")
    if item["status"] == "syntax_error":
        return _verify_syntax(item, manifest, hashes)
    info = json.loads(Path(item["report_path"]).read_text(encoding="utf-8"))[item["instance_id"]]
    if (not isinstance(info.get("resolved"), bool) or info.get("patch_successfully_applied") is not True
            or reward != float(info["resolved"])
            or item["status"] != ("resolved" if info["resolved"] else "unresolved")):
        raise RuntimeError("SWE-bench reward differs from the official resolved report")
    statuses = info["tests_status"]
    required = [name for group in ("FAIL_TO_PASS", "PASS_TO_PASS")
                for result in ("success", "failure") for name in statuses[group][result]]
    if not required or len(set(required)) != item["evaluated"]:
        raise RuntimeError("Official report test count differs from grading evidence")
    failures = statuses["FAIL_TO_PASS"]["failure"] + statuses["PASS_TO_PASS"]["failure"]
    if info["resolved"] != (not failures):
        raise RuntimeError("Official resolved flag differs from required test outcomes")
    item.update(officially_graded=True, changes=manifest["changes"])
    return item


class SWEBenchObservedTrainer(RepositoryObservedTrainer):
    """Reuse action/version and padding checks without toy-task grading assumptions."""

    def _evidence(self, values: dict) -> dict:
        evidence = AgentObservedTrainer._evidence(self, values)
        episodes, groups = [], {}
        for indices in episode_rows(values["rollout"].trajectories):
            row = values["rollout"].trajectories[indices[0]]
            item = verify_episode(row.metadata, float(row.reward), len(indices))
            item.update(prompt_id=row.prompt_id, truncated=row.truncated, terminal_reason=row.terminal_reason)
            episodes.append(item)
            groups.setdefault(row.prompt_id, []).append(float(row.reward))
        evidence.update(self.task_evidence)
        evidence.update(episodes=episodes, optimizer_steps=int(values["actor_update"].optimizer_steps),
                        mixed_reward_groups=[key for key, rewards in groups.items() if len(set(rewards)) > 1])
        return evidence


def summarize_acceptance(records: list[dict], require_uneven: bool = False) -> dict:
    """Require functional gates while reporting repair and learning separately."""
    if len(records) != 2:
        raise RuntimeError("SWE-bench acceptance requires exactly two completed steps")
    rows = []
    uneven = False
    for index, step in enumerate(records, start=1):
        if step["step"] != index or not step["ranks"]:
            raise RuntimeError("SWE-bench acceptance requires consecutive steps from policy zero")
        uneven = _validate_rank_alignment(step["ranks"]) or uneven
        for row in step["ranks"]:
            if (row["sampled_version"] != index - 1 or row["published_version"] != index
                    or row["optimizer_steps"] <= 0 or row["valid_tokens"] <= 0):
                raise RuntimeError("SWE-bench acceptance requires real optimizer execution and policy publication")
            if any(not math.isfinite(row[key]) for key in (
                    "gradient_norm", "loss", "parameter_max_delta", "task_advantage_max_abs")):
                raise RuntimeError("SWE-bench update evidence must be finite")
            rows.append(row)
    if require_uneven and not uneven:
        raise RuntimeError("SWE-bench acceptance requires actual unequal calls and zero-contribution padding")
    episodes = [episode for row in rows for episode in row["episodes"]]
    graded = [episode for episode in episodes if episode["officially_graded"]]
    if not any(episode["call_count"] > 1 for episode in graded):
        raise RuntimeError("No real multi-call submission reached the official grader")
    learning = any(row["mixed_reward_groups"] and row["task_advantage_nonzero_tokens"] > 0
                   and row["gradient_norm"] > 0 and row["parameter_max_delta"] > 0 for row in rows)
    return {"status": "passed", "functional_flow": "passed", "steps": len(records),
            "graded_episodes": len({episode["episode_id"] for episode in graded}),
            "uneven_calls_observed": uneven, "task_learning": "passed" if learning else "not_observed",
            "autonomous_repair": "passed" if any(episode["reward"] == 1 for episode in graded) else "not_observed"}


def main() -> None:
    """Run two-step functional acceptance through the production trainer with torchrun."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--require-uneven-calls", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if float(config["algorithm"].get("kl_coef", 0)) or float(config["train"]["optimizer"].get("weight_decay", 0)):
        raise ValueError("SWE-bench task-learning evidence requires zero KL and weight decay")
    trainer = SWEBenchObservedTrainer(config, args.output)
    rank = dist.get_rank()
    trainer.train()
    summary = summarize_acceptance(trainer.records, args.require_uneven_calls)
    step = trainer.records[-1]["step"]
    if trainer.evaluator is None or trainer.evaluator.last_step != step:
        raise RuntimeError("SWE-bench acceptance requires final-policy evaluation")
    checkpoint = trainer.checkpoints.directory(step) / "checkpoint_complete.json"
    metadata = json.loads(checkpoint.read_text(encoding="utf-8"))
    if metadata["step"] != step or metadata["world_size"] != len(trainer.records[-1]["ranks"]):
        raise RuntimeError("SWE-bench final checkpoint differs from observed step/world size")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "completed.json").write_text(json.dumps({
            **summary, "evaluation_step": step, "checkpoint": str(checkpoint), "checkpoint_reload": "not_run",
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
