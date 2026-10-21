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
"""CPU checks for portable experiment preparation and guarded dependency patches."""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

SCRIPTS = Path(__file__).resolve().parents[4] / "hyper_parallel/rl/examples/gsm8k/experiments/frozen_forward"


def load_tool(name: str) -> ModuleType:
    """Load standalone host utilities without importing Torch or starting hardware."""
    spec = importlib.util.spec_from_file_location(f"experiment_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFrozenForwardTools(unittest.TestCase):
    """Preserve experiment identity, incomplete evidence and exact sampler patch scope."""

    def setUp(self) -> None:
        """Load the sibling tools with isolated module bindings."""
        self.analyze = load_tool("analyze")
        with patch.dict(sys.modules, {"analyze": self.analyze}):
            self.monitor = load_tool("monitor")
        with patch.dict(sys.modules, {"monitor": self.monitor}):
            self.runner = load_tool("run")
        self.sampler = load_tool("patch_sampler")

    def test_configs_differ_only_in_forward_budgets_and_label(self) -> None:
        """Both cases retain one update, the same Actor and disabled validation/probability return."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SCRIPTS.parents[5]
            configs = [
                json.loads((self.runner.prepare_case(repo, root, name) / "config.yaml").read_text())
                for name in ("baseline", "dynamic")
            ]
            for config in configs:
                self.assertEqual(config["train"]["max_steps"], 1)
                self.assertEqual(config["train"]["max_tokens_per_micro_batch"], 4096)
                self.assertFalse(config["rollout"]["collect_log_probs"])
                self.assertFalse(config["evaluation"]["enabled"])
            for key in ("old_log_prob_max_tokens_per_micro_batch", "reference_log_prob_max_tokens_per_micro_batch"):
                self.assertIsNone(configs[0]["train"][key])
                self.assertEqual(configs[1]["train"][key], 4096)
                configs[1]["train"][key] = None
            configs[1]["logging"]["experiment_name"] = configs[0]["logging"]["experiment_name"]
            self.assertEqual(configs[0], configs[1])

    def test_retry_archives_readonly_children_without_deleting_evidence(self) -> None:
        """Moving the host-owned attempt directory does not require deleting container-owned children."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "baseline"
            previous.mkdir()
            child = previous / "rollouts"
            child.mkdir()
            (child / "rank.jsonl").write_text("retained evidence")
            child.chmod(0o555)
            self.runner.prepare_case(SCRIPTS.parents[5], root, "baseline")
            saved = list((root / "interrupted").glob("*/rollouts/rank.jsonl"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].read_text(), "retained evidence")
            saved[0].parent.chmod(0o755)

    def test_unknown_sampler_is_not_modified(self) -> None:
        """Changing dependencies cannot silently redirect the accepted fix to unknown code."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sampler.py"
            target.write_bytes(b"unknown implementation")
            with self.assertRaisesRegex(ValueError, "Unknown sampler"):
                self.sampler.apply_patch(target)
            self.assertEqual(target.read_bytes(), b"unknown implementation")

    def test_patch_is_exact_and_idempotent(self) -> None:
        """The accepted transformation only inserts stream lifetime tracking once."""
        original = b"def sample():\n    torch.npu.current_stream().wait_stream(global_stream())\n    return q\n"
        fixed = original.replace(b"    return q", b"    q.record_stream(torch.npu.current_stream())\n    return q")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sampler.py"
            target.write_bytes(original)
            with patch.object(self.sampler, "ORIGINAL_SHA256", hashlib.sha256(original).hexdigest()), patch.object(
                self.sampler,
                "PATCHED_SHA256",
                hashlib.sha256(fixed).hexdigest(),
            ):
                self.assertEqual(self.sampler.apply_patch(target)["status"], "applied")
                self.assertEqual(self.sampler.apply_patch(target)["status"], "already_applied")
            self.assertEqual(target.read_bytes(), fixed)

    def test_idle_check_rejects_missing_cards_foreign_work_and_query_failure(self) -> None:
        """A healthy memory reading alone cannot authorize a performance run."""
        devices = {str(i): {"capacity_mb": 65536, "hbm_mb": 4096} for i in range(8)}
        row = {"npu_smi_exit_code": 0, "foreign_compute": {}, "npu_smi": "fixture"}
        with patch.object(self.monitor, "cards", return_value=devices):
            self.assertTrue(self.monitor.idle(row))
            self.assertFalse(self.monitor.idle({**row, "foreign_compute": {"123": "external"}}))
            self.assertFalse(self.monitor.idle({**row, "npu_smi_exit_code": 124}))
        with patch.object(self.monitor, "cards", return_value={}):
            self.assertFalse(self.monitor.idle(row))

    def test_json_lines_keep_unicode_separators_inside_responses(self) -> None:
        """Only a real LF separates records in generated answer JSONL."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "answers.jsonl"
            target.write_text(json.dumps({"response": "a\u2028b"}, ensure_ascii=False) + "\n")
            self.assertEqual(self.analyze.records(target), [{"response": "a\u2028b"}])

    def test_case_cleanup_stops_only_owned_container_on_monitor_exit(self) -> None:
        """A dead monitor must stop the named case and never accept its output."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(results=root, wait_timeout=10)
            monitor, process = MagicMock(), MagicMock()
            monitor.__enter__.return_value = monitor
            process.__enter__.return_value = process
            monitor.poll.return_value = 1
            process.poll.return_value = None
            with patch.object(self.runner, "wait_idle"), patch.object(
                self.runner,
                "create_container",
                return_value="owned-container-id",
            ), patch.object(self.runner.subprocess, "Popen", side_effect=[monitor, process]), patch.object(
                self.runner.subprocess,
                "run",
            ) as stop:
                with self.assertRaisesRegex(RuntimeError, "Monitoring interrupted"):
                    self.runner.run_case(args, SCRIPTS.parents[5], "baseline")
                self.assertTrue(stop.call_args_list)
                for call in stop.call_args_list:
                    self.assertEqual(call.args[0], ["docker", "stop", "-t", "10", "owned-container-id"])
            self.assertTrue((root / "baseline/stop_monitor").exists())
            self.assertFalse((root / "baseline/result.json").exists())
