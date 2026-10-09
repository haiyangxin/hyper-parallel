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
"""Exact runtime combinations for Hyper model and numerical-profile adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VLLMRuntimeProfile:
    """Identify a reviewed API generation and its additional numerical dependencies."""

    name: str
    vllm_version: str
    ascend_version: str
    consistency_versions: tuple[tuple[str, str], ...] = ()


VLLM_ASCEND_0221 = VLLMRuntimeProfile("vllm_ascend_0221", "0.22.1", "0.22.1rc1")
VLLM_ASCEND_0230 = VLLMRuntimeProfile(
    "vllm_ascend_0230_post1", "0.23.0", "0.23.0.post1",
    (("torch", "2.10.0"), ("torch-npu", "2.10.0.post4")),
)
SUPPORTED_VLLM_RUNTIMES = (VLLM_ASCEND_0221, VLLM_ASCEND_0230)


def resolve_vllm_runtime(vllm_version: str, ascend_version: str) -> VLLMRuntimeProfile | None:
    """Return the exact adapter generation, rejecting unreviewed or mixed pairs."""
    pair = (vllm_version.split("+", maxsplit=1)[0], ascend_version.split("+", maxsplit=1)[0])
    for profile in SUPPORTED_VLLM_RUNTIMES:
        if pair == (profile.vllm_version, profile.ascend_version):
            return profile
    return None
