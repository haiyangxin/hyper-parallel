# Hyper-RL 运行镜像

普通 RL、Codex Agent 和 DeepSeek Harness 共用一个镜像，启动脚本和系统测试均默认使用此版本。

## 下载与校验

镜像公开可读，无需登录：

```bash
image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-unified-arm64
docker pull "${image}"
docker image inspect --format '{{index .RepoDigests 0}} {{.Os}}/{{.Architecture}}' "${image}"
```

| 项目            | 值                                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------- |
| 平台            | `linux/arm64`，Ascend NPU                                                                                         |
| Manifest digest | `sha256:450b32a4a1d7818d80b832335665a7d8bece5b91827f8ee80a9f62609e939944`                                         |
| 展开大小        | 约 18.9 GB                                                                                                          |
| 训练与推理      | Torch`2.10.0+cpu`、torch-npu `2.10.0`、Transformers `5.5.4`、vLLM `0.22.1+empty`、vLLM-Ascend `0.22.1rc1` |
| 数值一致性依赖  | `batch_invariant_ops==1.0.0`、`flash-attn-npu==0.2.0b1`                                                         |
| Agent           | Codex CLI`0.152.1`；DeepSeek Harness SDK/runtime `0.1.1rc1`                                                     |

Codex 使用独立二进制包，无需 Node.js。DeepSeek 指 Agent harness，不表示新增模型支持。
检查 Agent 依赖：

```bash
docker run --rm "${image}" /bin/bash -lc '
set -e
codex --version
python -c "import deepseek_harness; from importlib.metadata import version; print(version(\"deepseek-harness-sdk\"))"
'
```

## Qwen3.8 候选环境

Qwen3.8 的新环境以官方 `v0.23.0.post1` 为基础，保留原配 Torch、torch-npu、vLLM、Ascend、
Transformers 和 HTTP 依赖。它是独立验收中的候选环境，不替换上面的已验收统一镜像。
模型接入、旧模型回归和 RL 验收顺序见[接入开发文档](../docs/qwen3_8_development.md)。

基础镜像固定为 Linux ARM64 manifest
`quay.io/ascend/vllm-ascend@sha256:e62e85cb1bef9625cf568d0e8e2ac8e4b76f946eb6e8d9d459cfcf99565bea5d`。
[派生 Dockerfile](Dockerfile.qwen3_8)只增加上述受信镜像的 Codex CLI，以及固定的 DeepSeek SDK/runtime；
构建同时检查这些组件的 SHA256。源码、权重、日志与实验结果不打入镜像。

在用户指定的 reference 目录准备 SDK/runtime wheel；国内下载显式禁用代理：

```bash
wheel_dir=/home/xhy/Project-hw/reference/qwen3_8_env/wheels
mkdir -p "${wheel_dir}"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    NO_PROXY='*' no_proxy='*' python -m pip download --no-deps \
    --platform manylinux_2_28_aarch64 --python-version 312 --only-binary=:all: \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple --dest "${wheel_dir}" \
    deepseek-harness-sdk==0.1.1rc1 deepseek-harness-runtime-bin==0.1.1rc1
docker build --network none --pull=false -f hyper_parallel/rl/docker/Dockerfile.qwen3_8 \
    -t hyper-parallel/hyper-rl:qwen3_8-v0.23.0.post1-agent-arm64 "${wheel_dir}"
```

构建前需缓存两个固定来源镜像。官方源慢时，可显式禁用代理从国内 registry mirror 获取基础镜像，
核对 ARM64 manifest digest 后复用；`--network none` 只约束构建步骤，不保证 Docker daemon 拉取直连。
需要代理时使用宿主机当前端口 `8990`，只在对应进程作用域配置。

启动沿用现有镜像覆盖参数，挂载当前 checkout、模型和独立结果目录。
需要 editable 安装时，应使用可读取 checkout 的 UID，并在容器 passwd 中登记该 UID；
torch-npu 初始化会查询该身份。共享源码不可读时不要递归修改权限。
原配 custom vendor 算子库并非对所有 UID 可读；CPU editable 安装通过不代表该 UID 的
NPU 运行已通过。独立纯模型控制服务按官方 root 身份启动，并只挂载所需模型和空闲设备。
从仓库根目录安装 `python -m pip install --no-deps --no-build-isolation -e .`，
再安装 `python -m pip install --no-deps --no-build-isolation -e hyper_parallel/rl`；
检查两个导入路径和 `vllm.general_plugins` 的 `hyper_parallel` entry point。
只读源码启动时沿用 `install_runtime.sh` 的临时 RL 包安装方式，不对挂载源码执行 root editable 构建。

当前官方原配 `pip check` 含 profiler 缺库、HTTP/OpenCV 版本声明冲突和 Python 3.11 `te` wheel
在 Python 3.12 不受支持等失败；派生镜像保留这些 baseline 问题，不能声明全局依赖检查通过。
原配 FastAPI metrics 路由和 OpenCV 导入 smoke 通过，不等于所有 HTTP/多模态功能已验收。
RL 插件当前只为 0.22.1/0.22.1rc1 登记 Hyper 模型及私有生命周期/tool evidence 补丁；
在新环境中 entry point 可用且稳定 worker RPC 安装，不代表 Qwen3-4B bit-exact 或 Agent RL 已接通。
一致性算子、版本适配和真实 NPU 回归需要独立完成，不复制旧 ABI 的算子到新环境。

### Qwen3.8 native 服务启动

官方原配及 Agent 派生镜像均已独立完成两卡 TP2、BF16/eager 的真实文本生成预验。
下面沿用旧受信控制容器的
root 身份和驱动挂载，保留新镜像自己的 ENTRYPOINT 初始化 CANN、ascendnpu-ir 和 ATB。
只映射两张设备时，物理卡在容器内重新编号为 0/1，`ASCEND_RT_VISIBLE_DEVICES` 使用逻辑编号。
原配 custom vendor 目录为 root 所有且权限 750，改用任意非 root UID 会使算子库加载失败。

启动前检查 Health、计算进程与 `/proc/uda/namespace_node` 的驱动 namespace 占用；
下面的 4/5 是本次试验选择，不是固定资源预约。独立选择容器名、API/HCCL 端口，避免与其他作业冲突。
此入口只运行原生模型，不安装项目插件；权重发布和 RL 仍按开发文档独立验收。

```bash
image=hyper-parallel/hyper-rl:qwen3_8-v0.23.0.post1-agent-arm64
model_dir=/home/xhy/Project-hw/models/Qwen3.8-27B
device_a=4
device_b=5
docker run --rm --name hp-qwen3_8-native-smoke --user 0:0 --network host --shm-size 8g \
    --device "/dev/davinci${device_a}" --device "/dev/davinci${device_b}" \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "${model_dir}:/models/Qwen3.8-27B:ro" \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
    -e VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_HOST_IP=127.0.0.1 -e GLOO_SOCKET_IFNAME=lo \
    -e HCCL_IF_BASE_PORT=41000 -e HCCL_NPU_SOCKET_PORT_RANGE=42000-42999 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    "${image}" /bin/bash -c '
        set -euo pipefail
        unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
        export NO_PROXY="*" no_proxy="*"
        exec vllm serve /models/Qwen3.8-27B \
            --host 127.0.0.1 --port 18930 --served-model-name qwen3.8-27b \
            --tensor-parallel-size 2 --distributed-executor-backend mp \
            --dtype bfloat16 --enforce-eager --max-model-len 4096 \
            --max-num-seqs 1 --max-num-batched-tokens 512 --gpu-memory-utilization 0.7 \
            --logprobs-mode raw_logprobs --reasoning-parser qwen3
    '
```

设备枚举和库访问应在相同镜像、运行身份、ENTRYPOINT 及映射下预检；
CPU 安装、健康检查 HTTP 200 或其他 UID 的探针结果不能代替真实采样验收。

上述 `--enforce-eager` 用于功能预验。固定权重的 Qwen3.8 native 单请求对照中，
删除该参数并增加下面的选项后，真实全解码图编译/捕获/回放通过，暖态端到端输出速率
从约 5.29 提升到 28.75 token/s，其余 BF16、TP2、请求和采样设置保持相同：

```bash
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1}'
```

这里的单档捕获对应 `max-num-seqs=1`、无 MTP 的已验证配方；改变并发时另算捕获档位和内存。
图执行的初始化编译耗时单独记录，不能计入暖态速率或当作服务请求超时。
容器设备重编号会使当前自动 CPU 绑定按逻辑编号查找物理 topo 时失败；
本机仅手动绑定 worker 的短测没有明显额外提速，不为此改动宿主 IRQ 或其他进程。
这些结果仅覆盖固定权重 native 推理，动态发布后的图/权重/缓存生命周期仍需验收，
Qwen3-4B 原有 eager 与 bit-exact profile 的断言继续保留。

## 单轮 code 的 SandboxFusion 镜像

沙箱服务与训练容器分开运行，不需要分配 NPU。固定的 Python-only ARM64 镜像已上传到同一 SWR 组织；
镜像公开可读，无需登录；已使用空凭证 Docker 配置验证匿名拉取及镜像身份。

```bash
sandbox_image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/sandboxfusion-python:v1-arm64
docker pull "${sandbox_image}"
docker image inspect --format '{{.Id}} {{index .RepoDigests 0}} {{.Os}}/{{.Architecture}}' "${sandbox_image}"
```

| 项目 | 值 |
| --- | --- |
| 平台 | `linux/arm64`，Python-only 沙箱 |
| Manifest digest | `sha256:2ceb45f3e0a86ced184339573ab2be8ebf7d450b9cb741da405cbab11f462a73` |
| Image ID | `sha256:1ac76247ee612e1ac4a07ae10192056f061d93438f7c16578c7b169938b24131` |
| 展开大小 | 约 18.54 GB；继承训练基础层，新增层展开大小约 113 MB |

上传已复用 23 个不同的已有层，仅上传 14 个新增层。拉取时 Docker 同样复用本地已有层，无需手工拆分或拼接；
没有缓存基础层的机器仍需下载缺失层。展开大小不等于压缩后的网络流量。
`agentic.code.runtime_version` 沿用上面的 Image ID，不能与 Manifest digest 混用。
启动方式、cgroup 与隔离要求见 [code 示例](../examples/code/README.md#sandboxfusion-部署)。
普通 CPU UT 不需要此镜像；真实沙箱 ST 和 code 训练才需要启动服务。

## 宿主要求

以下要求针对训练容器；沙箱的隔离权限与 cgroup 条件见上述 code 部署说明。

- Linux ARM64、Docker、兼容的 Ascend NPU driver；`npu-smi info` 正常。
- 按运行场景准备空闲、健康的 NPU，模型和数据目录，以及至少 30 GB Docker 可用空间。
- 使用匹配的仓库源码；代码、模型、数据和结果由启动脚本挂载，不打入此镜像。

容器入口加载 CANN，启动脚本调用 `docker/install_runtime.sh` 安装 RL 包并注册 vLLM 插件。
`docker/patches/vllm-dp-coordinator-timeout.patch` 将 vLLM DP Coordinator 的启动等待时间从 30 秒延长到 120 秒；
GSM8K 的 TP 与一致性启动脚本会在容器中应用它。更新 vLLM 后应先确认补丁仍适用。
具体运行步骤见 [Hyper-RL README](../README.md)，系统测试见 [ST 指南](../README.md#系统测试)。
需要自定义镜像时，保留各启动脚本的 `HYPER_*_IMAGE` 和测试的 `RL_ST_*_IMAGE` 覆盖方式。
