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
"""Exact runtime pairs retain the same Qwen3 numerical contract."""
# White-box checks inspect dependency validation state without installing NPU packages.
# pylint: disable=protected-access

import unittest
from unittest.mock import patch

from rl.consistency import qwen3_dense
from rl.consistency.runtime import VLLM_ASCEND_0221, VLLM_ASCEND_0230, resolve_vllm_runtime


class TestRuntimeProfiles(unittest.TestCase):
    """Validate complete dependency sets rather than independently allowed versions."""

    def test_runtime_pairs_reject_cross_generation_dependencies(self) -> None:
        """Local build suffixes pass, but mixed pairs and unknown releases do not."""
        self.assertIs(resolve_vllm_runtime("0.22.1+local", "0.22.1rc1"), VLLM_ASCEND_0221)
        self.assertIs(resolve_vllm_runtime("0.23.0+empty", "0.23.0.post1"), VLLM_ASCEND_0230)
        for pair in (("0.23.0", "0.22.1rc1"), ("0.22.1", "0.23.0.post1"), ("0.23.1", "0.23.0.post1")):
            with self.subTest(pair=pair):
                self.assertIsNone(resolve_vllm_runtime(*pair))

    def test_new_runtime_checks_npu_abi_and_preserves_profile_identity(self) -> None:
        """The new API generation additionally requires its matching Torch/NPU ABI."""
        versions = {**qwen3_dense._NUMERICAL_PACKAGE_VERSIONS, "vllm": "0.23.0+empty",
                    "vllm-ascend": "0.23.0.post1", "torch": "2.10.0+cpu", "torch-npu": "2.10.0.post4"}
        with patch.object(qwen3_dense, "package_version", side_effect=versions.__getitem__), \
                patch.object(qwen3_dense._runtime, "dependency_runtime", "unconfigured"):
            qwen3_dense._require_package_versions()
            self.assertEqual(qwen3_dense.consistency_runtime_state()["dependency_runtime"], VLLM_ASCEND_0230.name)
            self.assertEqual(qwen3_dense.consistency_profile({"consistency": {"enabled": True}}),
                             qwen3_dense.QWEN3_ASCEND_CONSISTENCY_V1)
            versions["torch-npu"] = "2.10.0"
            with self.assertRaisesRegex(ValueError, "torch-npu==2.10.0.post4"):
                qwen3_dense._require_package_versions()

    def test_new_runtime_keeps_exact_fa_and_batch_invariant_dependencies(self) -> None:
        """A numerically different FA kernel release must not pass the version gate."""
        versions = {**qwen3_dense._NUMERICAL_PACKAGE_VERSIONS, "vllm": "0.23.0", "vllm-ascend": "0.23.0.post1",
                    "torch": "2.10.0", "torch-npu": "2.10.0.post4", "flash-attn-npu": "0.4.0"}
        with patch.object(qwen3_dense, "package_version", side_effect=versions.__getitem__):
            with self.assertRaisesRegex(ValueError, "flash-attn-npu==0.2.0b1"):
                qwen3_dense._require_package_versions()
