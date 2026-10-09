#!/usr/bin/env bash
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

set -euo pipefail

# Run inside the target image after sourcing its default Ascend entrypoint.
reference_root=${1:?Pass the reference directory containing sources and wheels}
build_root=${2:?Pass a new build directory; existing objects must not be reused}
test ! -e "${build_root}"
mkdir -p "${build_root}" "${reference_root}/wheels"
cp -a "${reference_root}/sources/flash-attention-npu-c7528-c8948" "${build_root}/flash-attention-npu"
cp -a "${reference_root}/sources/batch-invariant-1.0.0/torch_ops_extension/batch_invariant_ops" \
    "${build_root}/batch-invariant"

# CANN 9.1 relocated basic headers and stopped exposing softmax declarations transitively.
python - "${build_root}/flash-attention-npu/setup.py" <<'PY'
from pathlib import Path
import sys

setup_path = Path(sys.argv[1])
source = setup_path.read_text()
old = '            os.path.join(ascend_home, "aarch64-linux/tikcpp/include"),\n'
if source.count(old) != 1:
    raise RuntimeError("Unexpected pinned FlashAttention include list")
new = old + '            os.path.join(ascend_home, "aarch64-linux/asc/impl/basic_api"),\n'
source = source.replace(old, new)
old = '            *ext.sources,\n'
if source.count(old) != 1:
    raise RuntimeError("Unexpected pinned FlashAttention compiler inputs")
new = (
    '            *(["-include", "kernel_operator.h", "-include",\n'
    '               f"{ascend_home}/aarch64-linux/asc/include/adv_api/activation/softmax.h", "-include",\n'
    '               f"{ascend_home}/aarch64-linux/asc/impl/adv_api/detail/activation/softmax/softmax_common/softmax_common_utils.h"]\n'
    '              if ext.name == "flash_attn_npu_2" else []),\n'
) + old
setup_path.write_text(source.replace(old, new))
PY

export MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_BUILD_VERSION=all
cd "${build_root}/flash-attention-npu"
python setup.py build_ext --parallel 2 build_py
python setup.py bdist_wheel --skip-build --dist-dir "${reference_root}/wheels"
cd "${build_root}/batch-invariant"
USE_NINJA=1 python setup.py bdist_wheel --dist-dir "${reference_root}/wheels"
