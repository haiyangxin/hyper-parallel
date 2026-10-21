# GSM8K 运行配置与脚本

## 选择入口

只改变部署方式、TP 大小或运行参数时，复用同一个入口并覆盖参数，不再复制完整 YAML。

| 脚本 | 用途与支持值 |
| --- | --- |
| [Agent 入口](scripts/run_qwen3_4b_agentic_docker.sh) | `HYPER_AGENTIC_TASK=gsm8k`（默认）、`gsm8k_multi`、`gsm8k_codex`、`gsm8k_deepseek`；两卡 Qwen3-4B |
| [TP 入口](scripts/run_qwen3_tp_docker.sh) | `colocated` / `disjoint`；Hyper / native、训练 TP 与推理 TP 可分别选择；不声明严格一致性 |
| [一致性入口](scripts/run_qwen3_consistency_docker.sh) | `colocated`（默认）/ `disjoint`；`HYPER_QWEN3_TP=1` / `2`，训推 TP 相同，开启严格一致性 |

三个脚本均支持在末尾追加 `--section.field=value`，可选值在配置字段及脚本参数旁注明。
不同入口使用各自的任务预算、设备配置和验收要求。
Agent 入口检查版本、checkpoint 和更新次数：internal 每步为 1 次，外部 harness 按实际调用行数允许多次更新。
例如选择 Codex 或 DeepSeek Harness，只改变 Agent 任务参数：

```bash
export HYPER_VLLM_MODEL_ROOT=/path/to/models/Qwen3-4B
export HYPER_GSM8K_DATA_ROOT=/path/to/data/gsm8k
export HYPER_VLLM_VISIBLE_DEVICES=0,1
export HYPER_AGENTIC_TASK=gsm8k_codex  # gsm8k | gsm8k_multi | gsm8k_codex | gsm8k_deepseek
bash hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_4b_agentic_docker.sh \
  --train.model_init_cpu_threads=8
```

DeepSeek 指 harness，不是模型；它使用 Qwen3-4B。选择 `gsm8k_deepseek` 时，使用独立的
`HYPER_DEEPSEEK_MODEL_ROOT`、`HYPER_DEEPSEEK_DATA_ROOT` 和 `HYPER_DEEPSEEK_VISIBLE_DEVICES`
指定同类资源。四卡 MoE Codex 的命令见下文。

## 保留的配置

| 配置 | 独立保留的原因 |
| --- | --- |
| [Qwen3-4B 基础 GRPO](configs/qwen3_4b_gsm8k_vllm_production.yaml) | 单轮、内部多轮、短程与一致性脚本共用 |
| [GSPO](configs/qwen3_4b_gsm8k_gspo.yaml) | 序列级损失与算法字段不同，不能只改 GRPO 的算法名 |
| [PPO](configs/qwen3_4b_gsm8k_ppo.yaml) | 独立 Critic 及其优化配置 |
| [模型奖励](configs/qwen3_4b_gsm8k_model_reward.yaml) | 独立 Reward Model 服务及评分配置 |
| [Codex](configs/codex_multi_turn.yaml) | Codex CLI 协议、提示与预算 |
| [DeepSeek Harness](configs/deepseek_multi_turn.yaml) | SDK 协议、提示与预算 |
| [Qwen3-30B-A3B 单轮](configs/qwen3_30b_a3b_gsm8k_vllm.yaml) | MoE 模型与 EP/EDP 拓扑 |
| [Qwen3-30B-A3B Codex](configs/qwen3_30b_a3b_codex_multi_turn.yaml) | 四卡 MoE Codex、多轮预算与可选 R3 |

单轮和 TP2 consistency 入口使用基础配置及脚本参数覆盖；短程入口保留短程预算。
内部多轮使用以下参数，无需独立 YAML。

### 内部多轮参数

在基础 GRPO 配置上覆盖以下字段即可；这是 internal runner，不调用 Codex：

```bash
torchrun --nproc_per_node=8 hyper_parallel/rl/train_rl.py \
  hyper_parallel/rl/examples/gsm8k/configs/qwen3_4b_gsm8k_vllm_production.yaml \
  --agentic.interaction_mode=multi_turn \
  --agentic.apply_chat_template=true \
  --agentic.max_turns=4 \
  --agentic.max_observation_tokens=256
```

该命令保留基础配置的八卡完整训练规模；两卡短程检查使用 Agent 入口的 `gsm8k_multi` 模式。
工具协议默认 `json_function_call`，超时 5 秒、并发 2、每轮最多调用 2 次。

## Qwen3-30B-A3B Codex 多轮与 R3

[四卡运行配置](configs/qwen3_30b_a3b_codex_multi_turn.yaml)使用 Qwen3-30B-A3B、Codex CLI 0.152.1 和 GRPO。
训练为 TP2、DP shard2、EP2、EDP shard2，native vLLM 为 TP4、DP1；保持 consistency off 和 EPLB off。
任务预算与 [Codex 配置](configs/codex_multi_turn.yaml)一致：每题 4 个候选、最多 2 次模型调用、
每调用最多 256 tokens、8192 上下文、两步训练。预算耗尽而未完成答案的候选记零奖励，保留采样动作。
该配方默认关闭评测和 checkpoint，用于两步功能检查，不代表完整训练或收敛配方。

### 环境和路径

使用已准备好的四卡 Ascend Docker 训练容器，运行时为 vLLM 0.23.0、vLLM-Ascend 0.23.0.post1；
可使用 [v0.23 Code Agent 镜像](../../docker/README.md#仓库型-code-agent-与-qwen35-家族)。
GSM8K 的默认镜像现已统一为该 v0.23 镜像；本四卡 MoE 配方仍使用下面的直接训练命令，
不使用固定两卡的 Qwen3-4B Agent 启动器。
容器需要以下挂载；只读取已有模型和数据，不自动下载：

| 内容 | 容器路径 |
| --- | --- |
| 当前仓库源码 | `/repo` |
| Qwen3-30B-A3B 权重与 tokenizer | `/models/Qwen3-30B-A3B` |
| GSM8K `train.parquet`、`test.parquet` | `/data/gsm8k` |
| 可写结果目录 | `/results` |

按 [Docker 宿主要求](../../docker/README.md#宿主要求)挂载兼容的 Ascend driver。
以下命令在容器内、仓库根目录执行；设备编号替换为已分配的四张卡：

```bash
cd /repo
source /usr/local/Ascend/ascend-toolkit/set_env.sh
git config --global --add safe.directory /repo
python -m pip install --no-deps --no-build-isolation -e .
bash hyper_parallel/rl/docker/install_runtime.sh
export PYTHONPATH=/repo/hyper_parallel/rl:/repo:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOST_IP=127.0.0.1
export GLOO_SOCKET_IFNAME=lo
export HCCL_IF_BASE_PORT=62300
export HCCL_NPU_SOCKET_PORT_RANGE=62300-62399
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'
codex --version
```

导入应指向 `/repo`，Codex 版本应为 0.152.1。示例中的 Git 配置只在该容器内生效。
训练 HCCL 端口为 62300–62399，配方中 vLLM 使用 62600–62699；同时运行其他任务时需分配不冲突的端口。

### 默认运行与显式开启 R3

不加覆盖参数时，R3 关闭，初始化线程数保持环境原值，vLLM sleep level 为 1：

```bash
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=62299 \
  hyper_parallel/rl/train_rl.py \
  hyper_parallel/rl/examples/gsm8k/configs/qwen3_30b_a3b_codex_multi_turn.yaml
```

开启 R3，并将每个训练 rank 的模型初始化线程数限制为 8：

```bash
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=62299 \
  hyper_parallel/rl/train_rl.py \
  hyper_parallel/rl/examples/gsm8k/configs/qwen3_30b_a3b_codex_multi_turn.yaml \
  --train.router_replay.mode=R3 \
  --train.model_init_cpu_threads=8
```

初始化结束后恢复原线程设置，训练进程仍为四个。
R3 不改变任务预算、奖励规则或 Codex 协议；本配方已设置所需的推理 TP4/DP1。
支持范围、路由采集与重放合同见 [MoE R3](../../docs/moe_code_agent.md#moe-r3)。
CPU 内存不足时可追加 `--rollout.vllm.sleep_level=2`，适用条件见
[推理生命周期](../../docs/vllm_rollout.md#policy-生命周期)。

## 其他现有脚本

GSM8K Docker 启动脚本可在命令末尾接收 `train_rl.py` 的 `--section.field=value` 覆盖参数，例如：

```bash
bash hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_4b_agentic_docker.sh \
  --train.model_init_cpu_threads=8 \
  --rollout.vllm.sleep_level=2
```

仍须按该脚本原要求配置镜像、模型、数据和设备环境变量；这里的 dense 示例不启用 R3。
TP 和 consistency 脚本需先给出原来的 `colocated` 或 `disjoint` 位置参数，再追加覆盖参数。
覆盖参数必须指向 YAML 中已有字段；不传参数时保持原行为。Search-R1 的宿主启动器也支持相同形式的训练参数，
SWE-Smith 与单轮 code 的直接训练命令则沿用 `train_rl.py` 接口。

验收入口及边界见 [功能与验证说明](../../docs/moe_code_agent.md#系统测试)。

## 与 verl 的单步对比

[对比入口](scripts/run_verl_comparison_docker.sh)读取已完成的 verl 128 题 × 8 回答实验，
从原始 parquet 找出同一批题目，再从基础 GRPO 配方生成 `/results/config.yaml`。
使用本项目固定运行镜像，在 8 卡上训练 TP1、推理 TP2/DP4，完成一次更新及权重同步，默认关闭验证。

```bash
export VERL_BASELINE_RUN=/absolute/path/to/verl/completed-run
bash hyper_parallel/rl/examples/gsm8k/scripts/run_verl_comparison_docker.sh
```

基准目录须包含 `rollouts/1.jsonl`。可通过 `HYPER_COMPARISON_MODEL`、`HYPER_COMPARISON_DATA`、
`HYPER_COMPARISON_RESULTS` 指定模型、原始 GSM8K 数据和结果目录。
生成配置设置每 rank 16 题、每题 8 回答、每 rank mini batch 128 条、一次更新、1024/1024 长度、
Actor 重算 old log-prob、FP32 主参数、全参数 weight decay 0.01、学习率 1e-6、完整重计算。
推理设置 `rollout.collect_log_probs=false`，省去采样概率返回；更新分母仍由 Actor 重算。
`rollout.seed=null`、`rollout.vllm.seed=0` 使用引擎级随机状态，不承诺改变调度后仍逐条复现回答。
`rollout.vllm.disable_log_stats=true` 关闭 vLLM 服务端逐 token 统计；训练日志仍保留，
诊断服务端排队、prefill、decode 指标时须改为 false，并单独记录该开销。
`rollout.vllm.task_queue_enable=2` 对齐 verl 推理侧的 Ascend 算子下发队列等级，仅作用于推理子进程。
默认不执行验证、不保存 checkpoint。配置保留 1319 题、greedy、单回答和 1024-token 上限的验证参数；
单独执行验证时须显式启用 `evaluation.enabled`，训练结束验证还须启用 `evaluation.at_end`。
`evaluation.before_train=true` 在训练前验证当前策略；`evaluation.every_n_steps=10` 可独立于 checkpoint
每 10 步验证。省略该间隔或设为 null 时沿用保存边界，设为 0 时关闭周期验证；末步验证不会重复。
精度对照可将环境设为 `gsm8k_comparison_strict`，让训练与验证均采用 verl 严格奖励，默认数值奖励不变。

训练沿用数值归一化 GSM8K 奖励；[对比环境](verl_reward.py)额外记录 verl 严格格式得分。
每 rank 的 JSONL 同时记录训练和验证回答，`phase` 用于区分，验证补齐行须按题目去重。
该对照配方默认采用共卡 CPU offload 和固定两条的 micro batch，verl 基准采用动态 token batch；
框架、运行时版本和这些执行差异均须随性能结果披露。单步验证不代表长期收敛。

### 训练角色驻留与梯度同步

Qwen3 dense 的共卡 TP1 GRPO 可设置 `train.accelerator.cpu_offload: false`，
让 Actor 分片、FP32 主参数和优化器状态常驻设备；设置 `reference_cpu_offload: true` 可单独保留 Reference offload。
省略 Reference 覆盖时沿用 `cpu_offload`，现有 offload 配方行为不变。
此模式要求关闭 consistency；Reward Model 共卡路径仍要求 CPU offload。

`train.gradient_sync_each_micro_batch: true` 保留逐 microbatch 归约，适合单独测量驻留策略的影响。
默认 false 时，无 offload 的 Actor 在 mini-batch 最后一次反向才同步；这会增加梯度驻留内存。
两者都按既定 mini-batch 更新优化器，不改变问题数或每题回答数。
实际可用容量须覆盖权重发布和推理唤醒阶段，不能仅参考训练进程的 allocator 峰值。
框架在权重接收端确认加载后、唤醒 KV cache 前再次释放闲置缓存，避免发布临时缓存挤占推理空间。

`train.accelerator.comm_fusion_zero_copy` 复用共享 FSDP 的 zero-copy 通信选项，省略/null 保留底层默认；
设备常驻分片配合 `comm_fusion: true` 可避免重复输入打包。设置为 false 可单独对照该项收益。

### Actor 动态 token 分批

Qwen3 dense TP1 GRPO、consistency off 且未启用蒸馏时，可设置：

```yaml
train:
  max_tokens_per_micro_batch: 3000
```

该值限制每卡每次 Actor 前反向的完整输入 token 数，包含 prompt、response 和批内 padding。
省略或设为 null 恢复固定 `micro_batch_size`；old log-prob、Reference 前向仍沿用固定条数。
动态模式保留原有问题数、回答数、optimizer mini-batch 成员及更新次数，兼容 CPU offload。
各卡通过拆分分组对齐前反向次数；单条序列超预算时显式报错，不截断训练数据。
调参时同时查看 `train/micro_batches_per_rank`、`train/padding_fraction`、Actor 耗时和显存峰值。
此模式复用共享样本选择器与现有矩形模型输入，没有启用跨样本 packed attention。

## 性能实验与远端运行

[实验工作区](experiments/README.md)提供可迁移的 old/reference 单步对照工具、源码快照、
设备监控和日志回读流程；[优化跟踪](experiments/optimization_tracking.md)汇总已验证收益、精度边界和待办。
该工具保留 CPU offload 和概率返回开关，只在所选实验配方中关闭推理概率返回。
历史50步结果早于采样修复；新动态前向的远端 NPU 验收仍待执行。
