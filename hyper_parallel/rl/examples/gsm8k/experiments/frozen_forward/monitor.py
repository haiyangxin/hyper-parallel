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

"""Host-side NPU telemetry and contention guard for one experiment container."""
import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

from analyze import cards


def sample(identity: str = "") -> dict:
    """Treat query errors and unreadable live PIDs as unavailable hardware."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15, check=False)
        output, code = result.stdout, result.returncode
    except (OSError, subprocess.SubprocessError) as error:
        return {"utc": now, "npu_smi": "", "npu_smi_exit_code": 124, "foreign_compute": {}, "error": str(error)}
    owners = {}
    for line in output.split("Process id", 1)[-1].splitlines():
        parts = line.split("|")
        if len(parts) <= 3 or not parts[2].strip().isdigit():
            continue
        pid = parts[2].strip()
        try:
            cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
            owners[pid] = "owned" if identity and identity in cgroup else "external"
        except FileNotFoundError:
            continue
        except OSError:
            owners[pid] = "unknown"
    return {
        "utc": now,
        "npu_smi": output,
        "npu_smi_exit_code": code,
        "foreign_compute": {pid: owner for pid, owner in owners.items() if owner != "owned"},
    }


def idle(row: dict) -> bool:
    """Require eight readable cards, no external compute PIDs and 54 GiB free per card."""
    devices = cards(row["npu_smi"])
    return (
        row["npu_smi_exit_code"] == 0
        and not row["foreign_compute"]
        and len(devices) == 8
        and all(value["capacity_mb"] - value["hbm_mb"] >= 54 * 1024 for value in devices.values())
    )


def main() -> None:
    """Stop only the supplied experiment container when monitoring cannot ensure an uncontended run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    args = parser.parse_args()
    while not (args.case_dir / "stop_monitor").exists():
        row = sample(args.container_id)
        with (args.results / "device-monitor.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        reason = None
        if (args.results / "PAUSE").exists():
            reason = "user_pause"
        elif row["npu_smi_exit_code"] or row["foreign_compute"] or len(cards(row["npu_smi"])) != 8:
            reason = "external_occupancy_or_monitor_failure"
        if reason:
            (args.case_dir / "interruption.json").write_text(json.dumps({"reason": reason, "sample": row}, indent=2))
            subprocess.run(["docker", "stop", "-t", "10", args.container_id], check=True)
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
