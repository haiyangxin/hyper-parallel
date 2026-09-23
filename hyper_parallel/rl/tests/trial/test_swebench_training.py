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
"""SWE-bench training acceptance binds rewards to actual frozen grader artifacts."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import _swebench_train as swebench_train
from _swebench_train import summarize_acceptance, verify_episode
from test_repository_training import _records

from examples.code_agent.swebench_artifacts import _hash
from tests.common.mark_utils import arg_mark


_MARK = arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
                 card_mark="onecard", essential_mark="essential")


def _artifact(root: Path) -> dict:
    files = {"src/pkg.py": b"value = 1\n"}
    modes = {"src/pkg.py": 0o644}
    with tarfile.open(root / "submission.tar", "w") as stream:
        item = tarfile.TarInfo("src/pkg.py")
        item.size, item.mode = len(files[item.name]), modes[item.name]
        stream.addfile(item, io.BytesIO(files[item.name]))
    patch = b"example patch bytes"
    (root / "submission.patch").write_bytes(patch)
    manifest = {"content_hash": _hash(files, modes), "patch_sha256": hashlib.sha256(patch).hexdigest(),
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
                "modes": modes, "changes": {"added": [], "modified": ["src/pkg.py"], "deleted": []}}
    path = root / "submission.manifest.json"
    path.write_text(json.dumps(manifest))
    report = root / "submission.report.json"
    report.write_text(json.dumps({"case": {"resolved": False, "patch_successfully_applied": True,
                                         "tests_status": {"FAIL_TO_PASS": {"success": [], "failure": ["f"]},
                                                          "PASS_TO_PASS": {"success": ["p"], "failure": []}}}}))
    return {"episode_id": "episode", "instance_id": "case", "status": "unresolved", "artifact": str(path),
            "artifact_hash": manifest["content_hash"], "patch_sha256": manifest["patch_sha256"],
            "report_path": str(report), "evaluated": 2}


@_MARK
def test_official_artifact_checks_real_bytes_patch_and_report(tmp_path: Path) -> None:
    """Tampering at any artifact boundary must reject otherwise valid zero rewards."""
    metadata = _artifact(tmp_path)
    assert verify_episode(metadata, 0., 2)["officially_graded"]
    for key, value in (("artifact_hash", "wrong"), ("patch_sha256", "wrong"), ("evaluated", 1),
                       ("status", "resolved")):
        with pytest.raises(RuntimeError):
            verify_episode({**metadata, key: value}, 0., 2)
    with pytest.raises(RuntimeError, match="resolved report"):
        verify_episode(metadata, 1., 2)
    (tmp_path / "submission.patch").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="patch SHA"):
        verify_episode(metadata, 0., 2)
    metadata = _artifact(tmp_path)
    with tarfile.open(tmp_path / "submission.tar", "w"):
        pass
    with pytest.raises(RuntimeError, match="frozen artifact"):
        verify_episode(metadata, 0., 2)


@_MARK
def test_resolved_reward_matches_complete_official_report(tmp_path: Path) -> None:
    """A real positive must agree with required test outcomes and binary reward."""
    metadata = _artifact(tmp_path)
    report_path = Path(metadata["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["case"]["resolved"] = True
    metadata["status"] = "resolved"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="required test outcomes"):
        verify_episode(metadata, 1., 2)
    report["case"]["tests_status"]["FAIL_TO_PASS"] = {"success": ["f"], "failure": []}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert verify_episode(metadata, 1., 2)["officially_graded"]


@_MARK
def test_syntax_zero_binds_compiler_evidence_to_frozen_source(tmp_path: Path) -> None:
    """Syntax negatives remain outside official grading and reject mismatched evidence."""
    metadata = _artifact(tmp_path)
    manifest = json.loads(Path(metadata["artifact"]).read_text(encoding="utf-8"))
    compiler = {"python": "3.9.23", "executable": "/opt/miniconda3/envs/testbed/bin/python"}
    evidence = {"artifact_hash": metadata["artifact_hash"], "patch_sha256": metadata["patch_sha256"],
                "baseline": {**compiler, "files": {"src/pkg.py": "base hash"}, "errors": []},
                "candidate": {**compiler, "files": manifest["files"],
                              "errors": [{"path": "src/pkg.py", "type": "IndentationError", "lineno": 1,
                                          "offset": 1, "message": "unexpected indent"}]}}
    path = tmp_path / "submission.syntax.json"
    path.write_text(json.dumps(evidence))
    metadata.update(status="syntax_error", syntax_path=str(path), failure_origin="model", trainable=True, evaluated=0)
    result = verify_episode(metadata, 0., 2)
    assert result["syntax_verified"] and not result["officially_graded"]
    with pytest.raises(RuntimeError, match="syntax evidence"):
        verify_episode(metadata, 1., 2)
    evidence["candidate"]["files"]["src/pkg.py"] = "changed"
    path.write_text(json.dumps(evidence))
    with pytest.raises(RuntimeError, match="syntax evidence"):
        verify_episode(metadata, 0., 2)


@_MARK
def test_ungraded_zero_requires_explicit_submission_or_model_failure() -> None:
    """Infrastructure failures must never masquerade as trainable unresolved tasks."""
    for metadata in ({"status": "invalid_submission", "reason": "protected file"},
                     {"failure_origin": "model", "trainable": True}):
        assert not verify_episode(metadata, 0., 2)["officially_graded"]
        with pytest.raises(RuntimeError):
            verify_episode(metadata, 1., 2)
    for metadata in ({}, {"status": "unresolved"}, {"failure_origin": "infrastructure", "trainable": True}):
        with pytest.raises(RuntimeError, match="explicit"):
            verify_episode(metadata, 0., 2)


def _swe_records() -> list[dict]:
    records = _records()
    for step in records:
        for rank in step["ranks"]:
            for episode in rank["episodes"]:
                episode.update(officially_graded=True, evaluated=16, status="unresolved")
    return records


@_MARK
def test_functional_zero_rewards_do_not_imply_learning() -> None:
    """Real optimizer execution and publications can pass with no learning signal."""
    records = _swe_records()
    result = summarize_acceptance(records, require_uneven=True)
    assert result["functional_flow"] == "passed"
    assert result["task_learning"] == result["autonomous_repair"] == "not_observed"
    for override in ({"optimizer_steps": 0}, {"sampled_version": 0}, {"gradient_norm": float("nan")},
                     {"padding_rows": 0}):
        changed = deepcopy(records)
        changed[1]["ranks"][0].update(override)
        with pytest.raises(RuntimeError):
            summarize_acceptance(changed)
    with pytest.raises(RuntimeError, match="two completed"):
        summarize_acceptance(records[:1])
    for step in records:
        for rank in step["ranks"]:
            for episode in rank["episodes"]:
                episode["officially_graded"] = False
    with pytest.raises(RuntimeError, match="official grader"):
        summarize_acceptance(records)


@_MARK
def test_main_writes_completion_after_training_destroys_process_group(tmp_path: Path, monkeypatch) -> None:
    """Capture rank before cleanup; final reporting must not access the destroyed group."""
    config = tmp_path / "config.yaml"
    config.write_text("algorithm: {kl_coef: 0}\ntrain: {optimizer: {weight_decay: 0}}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint_complete.json"
    checkpoint.write_text(json.dumps({"step": 2, "world_size": 2}), encoding="utf-8")
    alive = [True]

    def rank() -> int:
        """Simulate the distributed rank API's lifecycle requirement."""
        if not alive[0]:
            raise RuntimeError("Default process group has been destroyed")
        return 0

    trainer = SimpleNamespace(records=_swe_records(), evaluator=SimpleNamespace(last_step=2),
                              checkpoints=SimpleNamespace(directory=lambda _step: tmp_path),
                              train=MagicMock(side_effect=lambda: alive.__setitem__(0, False)))
    monkeypatch.setattr(swebench_train, "SWEBenchObservedTrainer", lambda *_args: trainer)
    monkeypatch.setattr(swebench_train.dist, "get_rank", rank)
    output = tmp_path / "acceptance"
    monkeypatch.setattr("sys.argv", ["_swebench_train.py", str(config), str(output), "--require-uneven-calls"])
    swebench_train.main()
    result = json.loads((output / "completed.json").read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["evaluation_step"] == 2
    assert result["checkpoint"] == str(checkpoint)
    assert not alive[0]
