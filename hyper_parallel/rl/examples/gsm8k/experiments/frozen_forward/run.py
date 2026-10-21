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
"""Run a resumable pair of single-step experiments from one committed source snapshot."""
import argparse
from contextlib import ExitStack
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from monitor import idle, sample

IMAGE = "swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64"
SCRIPTS = Path("hyper_parallel/rl/examples/gsm8k/experiments/frozen_forward")


def command(args: list[str], **kwargs: object) -> str:
    """Run a command with explicit arguments and return captured standard output."""
    return subprocess.check_output(args, text=True, **kwargs).strip()


def write_status(results: Path, stage: str, **details: object) -> None:
    """Publish a status that remains readable while training writes large logs."""
    temporary = results / "progress.tmp"
    temporary.write_text(
        json.dumps({"stage": stage, "time": datetime.datetime.now().astimezone().isoformat(), **details}, indent=2)
        + "\n"
    )
    temporary.replace(results / "progress.json")


def snapshot(repo: Path, results: Path, image: str, data: Path, model: Path) -> Path:
    """Keep training inputs fixed while the development checkout changes independently."""
    if command(["git", "status", "--porcelain"], cwd=repo):
        raise ValueError(
            "Commit source changes before running; use a clean checkout and an ignored/external results path"
        )
    manifest = {
        "commit": command(["git", "rev-parse", "HEAD"], cwd=repo),
        "image_id": command(["docker", "image", "inspect", image, "--format", "{{.Id}}"]),
        "data": str(data),
        "model": str(model),
        "train_sha256": hashlib.sha256((data / "train.parquet").read_bytes()).hexdigest(),
    }
    record = results / "run_manifest.json"
    source = results / "source"
    if record.exists():
        if json.loads(record.read_text()) != manifest:
            raise ValueError("Run identity changed; use a new results directory")
        if not (source / SCRIPTS / "run_inside.sh").exists():
            raise ValueError("Incomplete source snapshot; use a new results directory")
        return source
    source.mkdir()
    archive = results / "source.tar"
    subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), "HEAD"], cwd=repo, check=True)
    subprocess.run(["tar", "-xf", str(archive), "-C", str(source)], check=True)
    record.write_text(json.dumps(manifest, indent=2) + "\n")
    return source


def wait_idle(results: Path, name: str, timeout: int) -> None:
    """Wait for three idle observations; never change another user's processes or device-share setting."""
    deadline = time.monotonic() + timeout
    consecutive = 0
    while consecutive < 3:
        if (results / "PAUSE").exists():
            raise InterruptedError("Paused by PAUSE file")
        if time.monotonic() >= deadline:
            raise TimeoutError("No idle device window before wait timeout")
        row = sample()
        with (results / "device-monitor.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        consecutive = consecutive + 1 if idle(row) else 0
        write_status(results, "waiting", case=name, idle_samples=consecutive)
        time.sleep(3)
    for device in range(8):
        value = command(["npu-smi", "info", "-t", "device-share", "-i", str(device), "-c", "0"])
        if "False" not in value:
            raise ValueError("Set device_share=false before this comparison")


def prepare_case(source: Path, results: Path, name: str) -> Path:
    """Archive incomplete attempts without deleting root-owned output files."""
    folder = results / name
    if folder.exists():
        attempts = results / "interrupted"
        attempts.mkdir(exist_ok=True)
        folder.rename(attempts / f"{name}-{time.time_ns()}")
    folder.mkdir()
    config = json.loads((source / SCRIPTS / "config.json").read_text())
    budget = None if name == "baseline" else 4096
    config["train"].update(
        old_log_prob_max_tokens_per_micro_batch=budget, reference_log_prob_max_tokens_per_micro_batch=budget
    )
    config["logging"]["experiment_name"] = f"frozen_forward_{name}"
    (folder / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    return folder


def create_container(args: argparse.Namespace, source: Path, folder: Path, name: str) -> str:
    """Mount explicit inputs into the standard image; preserve the caller's host environment."""
    mounts = [
        (source, "/repo", "rw"),
        (folder, "/results", "rw"),
        (args.model, "/models/Qwen3-4B", "ro"),
        (args.data, "/data", "ro"),
    ]
    mounts += [
        (Path(path), path, "ro")
        for path in (
            "/usr/local/Ascend/driver",
            "/usr/local/dcmi",
            "/usr/local/bin/npu-smi",
            "/etc/ascend_install.info",
        )
    ]
    argv = [
        "docker",
        "create",
        "--name",
        f"hp-frozen-{os.getpid()}-{name}",
        "--privileged",
        "--network",
        "host",
        "--shm-size",
        "64g",
        "-w",
        "/repo",
    ]
    for host, target, mode in mounts:
        argv.extend(["-v", f"{host}:{target}:{mode}"])
    for value in (
        "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
        "GLOO_SOCKET_IFNAME=lo",
        "HF_HUB_OFFLINE=1",
        "HF_DATASETS_OFFLINE=1",
        "VLLM_WORKER_MULTIPROC_METHOD=spawn",
    ):
        argv.extend(["-e", value])
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        argv.extend(["-e", key + "="])
    argv.extend([args.image, "bash", str(Path("/repo") / SCRIPTS / "run_inside.sh")])
    identity = command(argv)
    (folder / "container_id.txt").write_text(identity + "\n")
    return identity


def run_case(args: argparse.Namespace, source: Path, name: str) -> None:
    """Run one case with a separate host monitor and stop on any incomplete execution."""
    folder = prepare_case(source, args.results, name)
    wait_idle(args.results, name, args.wait_timeout)
    identity = create_container(args, source, folder, name)
    with ExitStack() as stack:
        monitor = None
        process = None
        try:
            with (folder / "monitor.log").open("w") as log:
                monitor = stack.enter_context(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(source / SCRIPTS / "monitor.py"),
                            "--container-id",
                            identity,
                            "--results",
                            str(args.results),
                            "--case-dir",
                            str(folder),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            write_status(args.results, "running", case=name, container_id=identity)
            with (folder / "launcher.log").open("w") as log:
                process = stack.enter_context(
                    subprocess.Popen(["docker", "start", "-a", identity], stdout=log, stderr=subprocess.STDOUT)
                )
                while process.poll() is None:
                    if monitor.poll() is not None or (folder / "interruption.json").exists():
                        subprocess.run(["docker", "stop", "-t", "10", identity], check=True)
                        raise RuntimeError("Monitoring interrupted the case; inspect interruption.json/monitor.log")
                    time.sleep(2)
            if process.returncode or (folder / "interruption.json").exists():
                raise RuntimeError(f"Case {name} did not finish successfully")
        finally:
            (folder / "stop_monitor").touch()
            subprocess.run(["docker", "stop", "-t", "10", identity], check=True)
            if process is not None:
                process.wait(timeout=30)
            if monitor is not None:
                monitor.wait(timeout=30)
    subprocess.run(
        [sys.executable, str(source / SCRIPTS / "analyze.py"), "--results", str(args.results), "--case", name],
        check=True,
    )


def main() -> None:
    """Resume successful cases without restarting them; never launch on a paused run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--wait-timeout", type=int, default=3600)
    args = parser.parse_args()
    args.model, args.data, args.results = [path.resolve() for path in (args.model, args.data, args.results)]
    if args.wait_timeout <= 0:
        parser.error("--wait-timeout must be positive")
    args.results.mkdir(parents=True, exist_ok=True)
    repo = Path(command(["git", "rev-parse", "--show-toplevel"]))
    with (args.results / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            source = snapshot(repo, args.results, args.image, args.data, args.model)
            for name in ("baseline", "dynamic"):
                if (args.results / "PAUSE").exists():
                    raise InterruptedError("Paused by PAUSE file")
                folder = args.results / name
                if (folder / "result.json").exists():
                    continue
                if (
                    (folder / "exit_code").exists()
                    and (folder / "exit_code").read_text().strip() == "0"
                    and not (folder / "interruption.json").exists()
                ):
                    subprocess.run(
                        [
                            sys.executable,
                            str(source / SCRIPTS / "analyze.py"),
                            "--results",
                            str(args.results),
                            "--case",
                            name,
                        ],
                        check=True,
                    )
                else:
                    run_case(args, source, name)
            write_status(args.results, "completed", report="performance_report.txt")
        except BaseException as error:
            write_status(args.results, "paused" if (args.results / "PAUSE").exists() else "failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
