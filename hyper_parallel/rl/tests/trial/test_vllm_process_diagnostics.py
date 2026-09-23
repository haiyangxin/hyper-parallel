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
"""Owned vLLM parent diagnostics distinguish observed exits from requested shutdown."""

import logging
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rl.roles.rollout import vllm
from tests.common.mark_utils import arg_mark


_MARK = arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
                 card_mark="onecard", essential_mark="essential")


@_MARK
def test_runtime_exit_keeps_last_live_resources_and_does_not_restart(caplog: pytest.LogCaptureFixture) -> None:
    """A real parent killed outside close is reported once and still fails the next request."""
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True) as process:
        client = vllm._VLLMHTTPClient(process, "http://unused", "test", 1)
        client._ready = True
        try:
            client._observe_process(sample_resources=True)
            last_alive = client._last_process_resources
            assert "sampled_at_utc" in last_alive
            client.start_process_monitor()
            process.terminate()
            process.wait(timeout=5)
            client._process_monitor.join(timeout=3)
            with pytest.raises(RuntimeError, match="exited during runtime"):
                client._request("GET", "health")
            evidence = client._exit_diagnostics
            assert evidence["return_code"] == -signal.SIGTERM
            assert evidence["parent_signal_number"] == signal.SIGTERM
            assert evidence["child_worker_signal"] == "unknown"
            assert evidence["phase"] == "runtime" and not evidence["shutdown_requested"]
            assert evidence["last_alive_resources"] is not None
            assert sum("vLLM process diagnostic:" in record.message for record in caplog.records) == 1
        finally:
            client.close()
        assert not client._process_monitor.is_alive()


@_MARK
def test_requested_close_records_parent_exit_and_stops_monitor(caplog: pytest.LogCaptureFixture) -> None:
    """Normal owned shutdown remains bounded and does not report an infrastructure crash."""
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True) as process:
        client = vllm._VLLMHTTPClient(process, "http://unused", "test", 1)
        client._ready = True
        client.start_process_monitor()
        with caplog.at_level(logging.INFO):
            client.close()
            client.close()
        assert client._exit_diagnostics["phase"] == "shutdown"
        assert client._exit_diagnostics["shutdown_requested"] is True
        records = [record for record in caplog.records if "vLLM process diagnostic:" in record.message]
        assert len(records) == 1 and records[0].levelno == logging.INFO
        assert not client._process_monitor.is_alive()


@_MARK
def test_parent_exit_one_does_not_invent_worker_signal_or_hide_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parent status 1 does not reveal how a child worker died, and network errors still propagate."""
    process = SimpleNamespace(pid=999999999, poll=Mock(return_value=None))
    client = vllm._VLLMHTTPClient(process, "http://unused", "test", 1)
    client._ready = True
    monkeypatch.setattr(vllm.urllib_request, "urlopen", Mock(side_effect=vllm.urllib_error.URLError("offline")))
    with pytest.raises(RuntimeError, match="offline"):
        client._request("GET", "health")
    assert client._exit_diagnostics is None
    process.poll.return_value = 1
    with pytest.raises(RuntimeError, match="code 1"):
        client._request("GET", "health")
    evidence = client._exit_diagnostics
    assert evidence["parent_signal_number"] is None and evidence["child_worker_signal"] == "unknown"
    assert evidence["exit_observation_resources"]["process_status"]["unavailable"] == "FileNotFoundError"


@_MARK
def test_unreadable_resource_files_preserve_original_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing diagnostic permissions must not replace the real owned-process exit."""
    monkeypatch.setattr(Path, "open", Mock(side_effect=PermissionError("not permitted")))
    client = vllm._VLLMHTTPClient(SimpleNamespace(pid=123, poll=lambda: -9), "http://unused", "test", 1)
    with pytest.raises(RuntimeError, match="code -9"):
        client._raise_if_owned_process_exited()
    resources = client._exit_diagnostics["exit_observation_resources"]
    assert all(value == {"unavailable": "PermissionError"}
               for key, value in resources.items() if key != "sampled_at_utc")


@_MARK
@pytest.mark.parametrize("failure", [OSError, RuntimeError])
def test_diagnostic_log_write_failure_does_not_replace_parent_exit(
    monkeypatch: pytest.MonkeyPatch, failure: type[Exception],
) -> None:
    """A broken log sink leaves the diagnostic object available and preserves the primary exit exception."""
    monkeypatch.setattr(vllm.logger, "log", Mock(side_effect=failure("log unavailable")))
    client = vllm._VLLMHTTPClient(SimpleNamespace(pid=999999999, poll=lambda: 1), "http://unused", "test", 1)
    with pytest.raises(RuntimeError, match="code 1"):
        client._raise_if_owned_process_exited()
    assert client._exit_diagnostics["return_code"] == 1


@_MARK
@pytest.mark.parametrize("membership,mountinfo,expected", [
    ("12:memory:/docker/owned", "20 19 0:43 /docker/owned /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory",
     (Path("/sys/fs/cgroup/memory"), False)),
    ("12:memory:/docker/owned/child", "20 19 0:43 /docker/owned /cg/mem rw - cgroup cgroup rw,memory",
     (Path("/cg/mem/child"), False)),
    ("0::/docker/owned/child", "20 19 0:43 /docker/owned /cg/unified rw - cgroup2 cgroup rw",
     (Path("/cg/unified/child"), True)),
    ("0::/owned group/child", r"20 19 0:43 /owned\040group /cg/with\040space rw - cgroup2 cgroup rw",
     (Path("/cg/with space/child"), True)),
    ("12:memory:/docker/other", "20 19 0:43 /docker/owned /cg/mem rw - cgroup cgroup rw,memory", None),
    ("0::/docker/owned", "20 19 0:43 /docker/owned /cg/mem rw - cgroup cgroup rw,memory", None),
])
def test_memory_cgroup_mount_root_mapping(membership: str, mountinfo: str, expected: object) -> None:
    """Non-root container mounts map only their own membership, including escaped mount paths."""
    assert vllm._VLLMHTTPClient._memory_cgroup_directory(membership, mountinfo) == expected
