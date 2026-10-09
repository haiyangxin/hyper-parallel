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
"""Launch the real single-NPU Qwen3.5/Qwen3.8 training contract."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


_WORKER = str(Path(__file__).resolve().with_name("_test_training.py"))
_FULL_WORKER = str(Path(__file__).resolve().with_name("_test_full_training.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_qwen35_training() -> None:
    """Run real hybrid-decoder gradients, padding, recomputation and recovery."""
    torchrun_case(file_name=_WORKER, case_name="test_qwen35_training", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_qwen35_fsdp_one_rank() -> None:
    """Verify the real one-rank FSDP lifecycle on a hybrid decoder."""
    torchrun_case(file_name=_WORKER, case_name="test_qwen35_fsdp_training", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_qwen35_fsdp_two_ranks() -> None:
    """Verify HCCL sharding, recomputation and restore on two NPU ranks."""
    torchrun_case(file_name=_WORKER, case_name="test_qwen35_fsdp_training", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level2",
          card_mark="allcards", essential_mark="unessential")
def test_qwen38_full_forward() -> None:
    """Probe the opt-in real 27B FSDP4 build/forward capacity window."""
    torchrun_case(file_name=_FULL_WORKER, case_name="test_qwen38_full_forward", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level2",
          card_mark="allcards", essential_mark="unessential")
def test_qwen38_full_training() -> None:
    """Require the opt-in real 27B FSDP4 training update without checkpoint export."""
    torchrun_case(file_name=_FULL_WORKER, case_name="test_qwen38_full_training", num_proc=4)
