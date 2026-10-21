#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--inside" ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export PYTHONPATH=/repo/hyper_parallel/rl:/repo:${PYTHONPATH:-}
    export HYPER_COMPARISON_OUTPUT=/results
    export TOKENIZERS_PARALLELISM=false
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=4
    cd /repo
    git config --global --add safe.directory /repo
    python -m pip install --no-deps --no-build-isolation -e .
    bash hyper_parallel/rl/docker/install_runtime.sh
    startup_patch=/repo/hyper_parallel/rl/docker/patches/vllm-dp-coordinator-timeout.patch
    if patch --dry-run -d /vllm-workspace/vllm -p1 < "$startup_patch" >/dev/null; then
        patch -d /vllm-workspace/vllm -p1 < "$startup_patch"
    elif ! patch --dry-run -R -d /vllm-workspace/vllm -p1 < "$startup_patch" >/dev/null; then
        echo "The documented vLLM startup patch is incompatible with this image." >&2
        exit 1
    fi
    python - <<'PY'
import importlib.metadata
import torch
import torch_npu
import hyper_parallel
print("checkout", hyper_parallel.__file__, flush=True)
if not hyper_parallel.__file__.startswith("/repo/"):
    raise RuntimeError("Editable installation does not point at the experiment checkout")
if torch.npu.device_count() != 8:
    raise RuntimeError("This comparison requires eight visible NPUs")
for package in ("torch", "torch-npu", "transformers", "vllm", "vllm-ascend"):
    print(package, importlib.metadata.version(package), flush=True)
PY
    python hyper_parallel/rl/examples/gsm8k/scripts/prepare_verl_comparison.py \
        --baseline /baseline --data /data/gsm8k --output /results
    set +e
    timeout --signal=TERM --kill-after=60s "${HYPER_COMPARISON_TIMEOUT:-3600}s" \
        python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
        --master_addr=127.0.0.1 --master_port=62399 \
        hyper_parallel/rl/train_rl.py /results/config.yaml 2>&1 | tee /results/train.log
    statuses=("${PIPESTATUS[@]}")
    printf '%s\n' "${statuses[0]}" > /results/exit_code
    exit "${statuses[0]}"
fi

repo_root=$(git rev-parse --show-toplevel)
workspace_root=$(dirname "$repo_root")
: "${VERL_BASELINE_RUN:?Set VERL_BASELINE_RUN to the completed verl result directory}"
result_root=${HYPER_COMPARISON_RESULTS:-$repo_root/hyper_parallel/rl/output/verl-comparison-$(date -u +%Y%m%dT%H%M%SZ)}
model_root=${HYPER_COMPARISON_MODEL:-$workspace_root/models/Qwen3-4B}
data_root=${HYPER_COMPARISON_DATA:-$workspace_root/data/gsm8k}
image=${HYPER_COMPARISON_IMAGE:-swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64}
container_name=${HYPER_COMPARISON_CONTAINER:-hyper-rl-verl-comparison-$$}
mkdir -p "$result_root"
git rev-parse HEAD > "$result_root/source_commit.txt"
git diff > "$result_root/source.patch"
docker image inspect "$image" --format '{{.Id}}' > "$result_root/image_id.txt"
docker run --rm --name "$container_name" --privileged --network host --shm-size 64g \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "$repo_root:/repo" -v "$result_root:/results" \
    -v "$model_root:/models/Qwen3-4B:ro" -v "$data_root:/data/gsm8k:ro" \
    -v "$VERL_BASELINE_RUN:/baseline:ro" -w /repo \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_HOST_IP=127.0.0.1 \
    -e GLOO_SOCKET_IFNAME=lo -e HCCL_IF_BASE_PORT=62400 \
    -e HCCL_NPU_SOCKET_PORT_RANGE=62400-62599 \
    -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
    -e HTTP_PROXY= -e HTTPS_PROXY= -e ALL_PROXY= \
    -e http_proxy= -e https_proxy= -e all_proxy= \
    "$image" bash /repo/hyper_parallel/rl/examples/gsm8k/scripts/run_verl_comparison_docker.sh --inside
