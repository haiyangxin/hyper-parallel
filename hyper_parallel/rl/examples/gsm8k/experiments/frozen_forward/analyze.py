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
"""Audit one update and compare phase timings with sampled whole-card telemetry."""
import argparse
import datetime
import json
import math
import re
import statistics
from pathlib import Path


def records(path: Path) -> list[dict]:
    """Read LF-delimited JSON without splitting embedded Unicode separators."""
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def cards(output: str) -> dict:
    """Parse per-card HBM and AICore readings from the host tool output."""
    result = {}
    current = None
    for line in output.split("Process id", 1)[0].split("\n"):
        match = re.match(r"\|\s*(\d+)\s+910B", line)
        if match:
            current = int(match[1])
        parts = line.split("|")
        if current is not None and len(parts) > 3 and re.search(r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:", parts[2]):
            numbers = [int(x) for x in re.findall(r"\d+", parts[3])]
            if len(numbers) == 5:
                result[str(current)] = {"aicore": numbers[0], "hbm_mb": numbers[3], "capacity_mb": numbers[4]}
    return result


def audit(root: Path, name: str) -> dict:
    """Accept only a complete single update with eight-rank and telemetry evidence."""
    folder = root / name
    if int((folder / "exit_code").read_text()) != 0:
        raise RuntimeError("Training did not exit successfully")
    training = []
    for line in (folder / "train.log").read_text().split("\n"):
        if " | rl.utils.monitoring | step=" in line:
            values = {key: float(value) for key, value in re.findall(r"([a-zA-Z0-9_/]+)=([^, ]+)", line)}
            training.append(values)
    if len(training) != 1:
        raise RuntimeError("Expected exactly one training update")
    values = training[0]
    if (
        values["step"] != 1
        or values["train/optimizer_steps"] != 1
        or values["policy/version"] != 1
        or values["rollout/sequence_count"] != 1024
        or values["train/gradient_norm"] <= 0
    ):
        raise RuntimeError("Update, policy version or response count mismatch")
    for key in ("train/gradient_norm", "train/total_loss", "reward/mean"):
        if not math.isfinite(values[key]):
            raise RuntimeError(f"Nonfinite {key}")
    events = [record for rank in range(8) for record in records(folder / f"phases-rank-{rank}.jsonl")]
    started, finished = [float((folder / filename).read_text()) for filename in ("process_started", "process_finished")]
    samples = []
    for row in records(root / "device-monitor.jsonl"):
        stamp = datetime.datetime.fromisoformat(row["utc"]).timestamp()
        if started <= stamp <= finished:
            if row["npu_smi_exit_code"] or row["foreign_compute"]:
                raise RuntimeError("Resource-monitor failure or external contention in measured run")
            samples.append((stamp, cards(row["npu_smi"])))
    if not samples:
        raise RuntimeError("No device monitor evidence")
    phases = {}
    for phase in ("old_log_prob", "reference", "actor_update", "weight_publish", "service_startup", "prepare_training"):
        rows = [row for row in events if row["phase"] == phase]
        if phase in ("old_log_prob", "reference", "actor_update", "weight_publish") and len(rows) != 8:
            raise RuntimeError(f"Incomplete phase {phase}")
        if not rows:
            continue
        if any(row["status"] != "completed" for row in rows):
            raise RuntimeError(f"Failed phase {phase}")
        data = {
            "max_rank_seconds": max(row["seconds"] for row in rows),
            "mean_rank_seconds": statistics.mean(row["seconds"] for row in rows),
            "max_peak_allocated_gib": max(row["peak_allocated"] for row in rows) / 2**30,
            "max_peak_reserved_gib": max(row["peak_reserved"] for row in rows) / 2**30,
            "per_rank": rows,
        }
        # Use the interval shared by all ranks to avoid attributing another phase's utilization.
        low, high = max(row["started"] for row in rows), min(row["finished"] for row in rows)
        phase_samples = [devices for stamp, devices in samples if low <= stamp <= high]
        data["utilization_shared_interval_samples"] = len(phase_samples)
        data["cards"] = {}
        for card in map(str, range(8)):
            readings = [devices[card] for devices in phase_samples if card in devices]
            data["cards"][card] = (
                None
                if not readings
                else {
                    "aicore_mean_percent": statistics.mean(v["aicore"] for v in readings),
                    "hbm_peak_mb": max(v["hbm_mb"] for v in readings),
                    "min_free_mb": min(v["capacity_mb"] - v["hbm_mb"] for v in readings),
                }
            )
        if phase in ("old_log_prob", "reference"):
            if any(row["nonfinite"] or row["rows"] != 128 for row in rows):
                raise RuntimeError("Invalid forward observations")
            data["input_tokens"] = sum(row["input_tokens"] for row in rows)
            data["padded_tokens"] = sum(row["padded_tokens"] for row in rows)
            data["calls_per_rank"] = [row["calls"] for row in sorted(rows, key=lambda row: row["rank"])]
        phases[phase] = data
    result = {
        "case": name,
        "training": values,
        "phases": phases,
        "process_seconds": finished - started,
        "monitor_samples": len(samples),
        "foreign_samples": 0,
        "whole_run_hbm_peak_mb": max(v["hbm_mb"] for _, devices in samples for v in devices.values()),
        "max_sample_gap_seconds": max((b[0] - a[0] for a, b in zip(samples, samples[1:])), default=0),
        "limitations": (
            "Single update; sampled telemetry can miss transient peaks. "
            "No validation set. Initial service startup is separate."
        ),
    }
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    return result


def report(root: Path) -> None:
    """Summarize accepted cases without pooling interrupted attempts."""
    results = {
        name: json.loads((root / name / "result.json").read_text())
        for name in ("baseline", "dynamic")
        if (root / name / "result.json").exists()
    }
    lines = [
        "old/reference动态token：两组各1 step，采样修复开启，rollout log-prob返回关闭。",
        "Reference CPU offload保留；Actor更新预算4096不变。无验证、checkpoint或额外消融。",
    ]
    if len(results) == 2:
        for phase in ("old_log_prob", "reference", "actor_update", "weight_publish"):
            a, b = [results[name]["phases"][phase] for name in ("baseline", "dynamic")]
            lines.append(
                f"{phase}: {a['max_rank_seconds']:.4f} -> {b['max_rank_seconds']:.4f} s; "
                f"耗时降低 {(1-b['max_rank_seconds']/a['max_rank_seconds'])*100:.2f}%; "
                f"allocated峰值 {a['max_peak_allocated_gib']:.3f} -> {b['max_peak_allocated_gib']:.3f} GiB"
            )
        for name, value in results.items():
            lines.append(
                f"{name}: old/ref有效输入tokens={value['phases']['old_log_prob']['input_tokens']}; "
                f"整卡HBM采样峰值={value['whole_run_hbm_peak_mb']} MiB; 监控样本={value['monitor_samples']}"
            )
    lines.extend(
        [
            "详细逐rank和逐卡显存/利用率见各result.json；短阶段无采样时标null，不填0。",
            "生成轨迹独立；需结合有效token数量解释差异。只验证一轮更新，不证明稳态整步或长期精度。",
        ]
    )
    (root / "comparison.json").write_text(json.dumps(results, indent=2))
    (root / "performance_report.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Analyze results on either the experiment host or the local log mirror."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--case", choices=("baseline", "dynamic"))
    args = parser.parse_args()
    root = args.results.resolve()
    if args.case:
        audit(root, args.case)
    report(root)


if __name__ == "__main__":
    main()
