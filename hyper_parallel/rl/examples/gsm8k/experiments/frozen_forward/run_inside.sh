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

set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/repo/hyper_parallel/rl:/repo:${PYTHONPATH:-}
set -u
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TASK_QUEUE_ENABLE=2
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_HOST_IP=127.0.0.1 HCCL_IF_BASE_PORT=62600 HCCL_NPU_SOCKET_PORT_RANGE=62600-62799
export VLLM_ASCEND_ENABLE_NZ=0 VLLM_SERVER_DEV_MODE=1 VLLM_BATCH_INVARIANT=0
export HYPER_COMPARISON_OUTPUT=/results
unset PYTORCH_NPU_ALLOC_CONF
cd /repo
python -m pip install --no-deps --no-build-isolation -e .
bash hyper_parallel/rl/docker/install_runtime.sh
experiment=/repo/hyper_parallel/rl/examples/gsm8k/experiments/frozen_forward
python "$experiment/patch_sampler.py" --record /results/sampler.json
startup_patch=/repo/hyper_parallel/rl/docker/patches/vllm-dp-coordinator-timeout.patch
if patch --dry-run -d /vllm-workspace/vllm -p1 < "$startup_patch" >/dev/null; then
    patch -d /vllm-workspace/vllm -p1 < "$startup_patch"
elif ! patch --dry-run -R -d /vllm-workspace/vllm -p1 < "$startup_patch" >/dev/null; then
    echo "Incompatible vLLM startup patch" >&2
    exit 1
fi
python - <<'PY'
from pathlib import Path
import hyper_parallel
if not hyper_parallel.__file__.startswith('/repo/'):
    raise RuntimeError('Editable installation does not point at the frozen checkout')
Path('/results/import_path.txt').write_text(hyper_parallel.__file__)
PY
set +e
date +%s > /results/process_started
timeout --signal=TERM --kill-after=20s 1800s \
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
    --master_addr=127.0.0.1 --master_port=62399 "$experiment/observe.py" /results/config.yaml \
    > /results/train.log 2>&1
result=$?
date +%s > /results/process_finished
printf '%s\n' "$result" > /results/exit_code
exit "$result"
