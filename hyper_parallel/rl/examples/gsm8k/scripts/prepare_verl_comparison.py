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
"""Prepare a fixed-question, one-step comparison from an actual verl run."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml


def prepare(baseline: Path, data: Path, output: Path) -> None:
    """Reuse the baseline's exact questions and write a resolved Hyper-RL recipe."""
    rows = [json.loads(line) for line in (baseline / "rollouts/1.jsonl").read_text().split("\n") if line]
    prompt_counts = Counter(row["input"] for row in rows)
    if len(prompt_counts) != 128 or set(prompt_counts.values()) != {8}:
        raise ValueError("Expected the completed 128-question, eight-response baseline")
    frame = pd.read_parquet(data / "train.parquet")
    by_prompt = {
        "user\n" + row["prompt"][0]["content"] + "\nassistant\n": index
        for index, row in frame.iterrows()
    }
    indices = [by_prompt[prompt] for prompt in prompt_counts]
    selected = frame.iloc[indices].copy()
    output.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output / "train.parquet", index=False)
    base = Path(__file__).resolve().parents[1] / "configs/qwen3_4b_gsm8k_vllm_production.yaml"
    config = yaml.safe_load(base.read_text())
    config["data"].update(
        train_path=str(output / "train.parquet"), max_prompt_length=1024,
        max_train_samples=None, shuffle=False, num_workers=0,
    )
    config["rollout"].update(
        num_return_sequences=8, max_new_tokens=1024, seed=None, top_k=-1, collect_log_probs=False,
    )
    config["rollout"]["vllm"].update(
        data_parallel_size=4, tensor_parallel_size=2, model_implementation="native",
        port=18500, seed=0, disable_log_stats=True, task_queue_enable=2,
        batch_invariant=False, enforce_eager=True, enable_chunked_prefill=False,
        enable_prefix_caching=True, gpu_memory_utilization=0.6,
        max_model_len=2048, max_num_seqs=128, max_num_batched_tokens=8192,
        server_hccl_if_base_port=62600, server_hccl_npu_socket_port_range="62600-62799",
    )
    config["rollout"]["vllm"].pop("kv_cache_memory_bytes", None)
    config["rollout"]["vllm"]["weight_sync"].update(strategy="full_gather", bucket_size_mb=4096)
    config["agentic"].update(module_path="examples.gsm8k.verl_reward", environment="gsm8k_comparison")
    config["evaluation"].update(
        enabled=False, at_end=False, max_samples=1319, batch_size=32,
        max_new_tokens=1024, temperature=0.0, top_p=1.0, top_k=-1, do_sample=False,
    )
    config["train"].update(
        max_steps=1, seed=42, prompt_batch_size=16, micro_batch_size=2,
        response_mini_batch_size=128, policy_update_epochs=1,
        old_log_prob_source="actor", model_init_cpu_threads=8,
    )
    config["train"]["accelerator"].update(
        dp_shard=8, tp=1, cpu_offload=True, activation_checkpoint="full",
    )
    config["train"]["optimizer"].update(
        lr=1.0e-6, weight_decay=0.01, fp32_main_params=True, no_decay_params=[],
    )
    config["train"]["checkpoint"].update(
        output_dir=str(output / "checkpoints"), save_steps=0, save_final=False, verify_reload=False,
    )
    config["logging"].update(
        backends=["console"], log_steps=1, log_samples=0,
        experiment_name="qwen3_4b_verl_128x8",
    )
    config["logging"]["wandb"]["mode"] = "disabled"
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "baseline": str(baseline), "source_dataset_indices": indices,
        "source_question_ids": [int(row["extra_info"]["index"]) for _, row in selected.iterrows()],
        "questions": len(selected), "responses_per_question": 8,
        "baseline_rollouts_sha256": hashlib.sha256((baseline / "rollouts/1.jsonl").read_bytes()).hexdigest(),
        "execution_differences": [
            "Hyper-RL offloads Actor, Reference and optimizer state; verl only offloads Reference.",
            "Hyper-RL uses fixed micro batches of two responses; verl uses dynamic token budgets.",
            "Each framework uses its own official runtime image and FSDP implementation.",
            "Training uses numeric-normalized rewards here and strict-format rewards in the verl baseline.",
            "Rollout uses engine-level RNG without per-trajectory seeds; Actor recomputation supplies the anchor.",
            "Rollout disables token statistics and uses TASK_QUEUE_ENABLE=2; trainer environment is inherited.",
            "The original 7473-row dataset is reduced to the exact 128 questions consumed by the baseline step.",
        ],
    }
    (output / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    """Parse comparison input and output paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.baseline, args.data, args.output)


if __name__ == "__main__":
    main()
