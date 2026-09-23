# 仓库 Code Agent：工作区与独立判题

已完成两卡 Qwen3-4B 的仓库 Agent RL 两步功能闭环：真实调用、冻结判题、训练消费、权重发布、评估和保存。
训练奖励全零，未观察到有效任务学习；同任务短评估为 1/2，不代表泛化或训练收益。SWE-bench 两例官方正反例与两卡两步功能训练已通过；训练及最终评估均为零分，未观察到修复或学习。
后续接口和阶段边界见[开发文档](../../docs/code_agent_development.md)。

2026-09-28 新增的外部 API 仅推理路径已用 Qwen3.8-27B 通过两可控仓库及两 SWE 实例；
该结果与上述历史训练分开记录，详见[当前分支验收](../../docs/code_agent_handoff.md#外部-api-仅推理验收2026-09-28)。

## 数据与奖励

`fixtures/` 包含 `merge_intervals`、`word_counts` 的公开缺陷代码和公开测试。
`tasks.json` 中隐藏输入/预期及参考修复仅供控制端使用，不复制进候选容器。
`prepare_data.py` 生成两行功能数据，使用既有 `PromptRecord`、`data.row_adapter` 合同；
prompt 只含公开任务，ground truth 仅含 fixture ID、基线 hash 和测试版本 hash。
该小数据集只用于功能验证，不是 benchmark 或独立评估集。

`RepositoryTask.prepare(workspace)` 复制构造时固定的公开基线字节。
候选可增改删 `src/*.py`；其余基线文件受保护，额外非源码文件也拒绝。
Python 禁止写字节码缓存；CLI 状态保存在工作区之外，不通过宽泛忽略规则隐藏提交修改。

`RepositoryTask.evaluate(archive)` 校验停止后导出的 tar，并记录内容 hash、文件变更清单、
完整镜像 ID、测试和判题器版本。每个隐藏用例创建全新无网络 grader，先验证解释器健康，
再执行固定入口；预期值由控制端比较，候选不能以输出 `passed=true` 自行声明成功。
每题所有 5 条用例均执行，全部通过奖励为 1，否则为 0；候选失败不跳过后续用例。

明确非法提交、错误答案、候选执行超时/输出超限为任务零分。
Docker 管理/传输故障、缺失解释器、停止失败与未知运行故障上抛，不转换成零分。
每条用例重新建容器，避免候选修改解释器、测试入口或残留后台进程影响后续用例。

## 环境与构建

仅控制端需要 Docker 管理权限和当前仓库的 RL Python 依赖。
candidate/grader 不挂载宿主目录、Docker socket、NPU 或隐藏数据；默认无网络，
1 CPU、1 GiB 内存、64 进程、cap-drop ALL 和 no-new-privileges。
控制容器若持有 Docker socket，应视为受信控制端；不能复用成候选工作区。

从仓库根目录离线构建：

```bash
docker build --network none --pull=false \
  -t hyper-parallel/hyper-code-agent:stage-a-unified-arm64 \
  -f hyper_parallel/rl/docker/Dockerfile.code-agent hyper_parallel/rl/docker
docker image inspect --format '{{.Id}}' hyper-parallel/hyper-code-agent:stage-a-unified-arm64
```

Dockerfile 固定本地统一训练镜像身份，复用已有 Codex CLI 0.152.1，不联网安装。
派生镜像清空旧工作区、使用中性登录 shell 与独立可写 HOME/CODEX_HOME，避免初始化训练设备。
此修改只作用于派生镜像；正常训练仍用原统一镜像及其 CANN 初始化。
运行配置使用上一步读取的完整 `sha256:...`，任务评分不接受可变镜像标签。

## Codex 仓库任务接线

### 外部 API 仅推理验收

`examples/code_agent/inference.py` 用于已部署模型 API 的仓库功能验收。它复用当前 Codex CLI
0.152.1、Gateway 的 Responses→Chat 转换、仓库双工具合同、`CandidateRelay`、`DockerWorkspace`，
以及原有 `RepositoryTask` / `SWEBenchTask` 的准备、冻结与独立评分。
这与直接调用新版 Codex 的独立模型能力实验不同；两种路径的结果必须分开记录。

仅推理模式由受信控制端显式选择。`CodexAgentProgram.run_inference()` 返回推理结果与评分，
不返回 `Trajectory`；`policy_version` 为空，表示没有框架发布的 Actor 策略身份。
Gateway 保留原始 CLI 请求、实际后端请求、真实结构化响应和 usage，不要求外部服务伪造
vLLM 的 token ID、逐 token logprob 或 Hermes 解析证据。模型工具参数仍须符合实际工具 schema，
服务错误、断连、未知格式及失败清理仍显式失败，不能转换成普通零奖励。

默认训练路径保持原始采样证据校验；训练入口拒绝仅推理记录。推理成功不证明梯度、权重同步、
GRPO 或恢复训练通过，也不将 GGUF 外部服务登记为新的训练模型或 rollout engine。
API 密钥由控制端从文件读取，只在后端 HTTP 鉴权时使用，不写入候选配置、提示或采样日志。

上下文窗口必须与实际服务容量一致。仅推理模式可不设置模型调用次数上限；未显式配置的采样参数
沿用后端服务默认值。请求、CLI 总执行及响应字节仍有显式运行边界；终止原因与结果分别记录，
不得把超时或服务超限记作模型正常完成。仓库任务的原有摘要与工具输出合同继续生效。

任务修改范围保持不变：可控仓库只接受 `src/*.py`，SWE 任务只接受约定目录中的 Python 源码。
不要把独立实验产生的 changelog 或测试改动静默过滤后冒充原始提交；不合规产物按任务合同判定。
功能验收至少区分 CPU 合同、真实容器/固定 CLI 合成验证，以及真实模型完成任务的独立评分。

推理配方分别为 [`external_api_inference.json`](configs/external_api_inference.json)（可控仓库）和
[`external_api_swebench_inference.json`](configs/external_api_swebench_inference.json)（SWE）。
它们是独立入口使用的 Codex 配置，不是 `train_rl.py` 的完整训练 YAML。
默认 CLI 窗口 131072、无调用次数上限，保留服务采样默认；CLI 总时限 3600 秒，单请求时限
1800 秒。运行前核对 API 实际容量和模型别名，调整固定镜像 ID 及控制端 registry 路径。

在已安装依赖、核对当前 checkout 的 editable 导入且可管理 Docker 的受信控制端执行：

```bash
export PYTHONPATH="$PWD/hyper_parallel/rl:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m examples.code_agent.prepare_data --output /results/inference-inputs/toys.parquet
python -m examples.code_agent.swebench_data --registry /results/registry/registry.json \
  --output /results/inference-inputs/swebench.parquet

python -m examples.code_agent.inference \
  --config hyper_parallel/rl/examples/code_agent/configs/external_api_inference.json \
  --data /results/inference-inputs/toys.parquet --fixture-id word_counts \
  --backend-url http://127.0.0.1:18000 --backend-key-file /run/secrets/model-api-key \
  --model qwen3.8-27b --output /results/inference-word-counts

python -m examples.code_agent.inference \
  --config hyper_parallel/rl/examples/code_agent/configs/external_api_swebench_inference.json \
  --data /results/inference-inputs/swebench.parquet --instance-id pytest-dev__pytest-10051 \
  --backend-url http://127.0.0.1:18000 --backend-key-file /run/secrets/model-api-key \
  --model qwen3.8-27b --output /results/inference-pytest-10051
```

`--backend-url` 是不带 `/v1` 的服务根地址；密钥文件必须位于控制端，不能挂入 candidate/grader。
将 fixture 改为 `merge_intervals`、SWE instance 改为 `pytest-dev__pytest-10081`，并各自使用新输出目录，
可完成其余两个固定任务。已有输出目录拒绝覆盖。检查 `result.json` 中的独立评分与 session
下的真实调用、冻结产物和评分报告；`failure.json` 表示未完成，不应视作正常零分。

### 训练配置中的仓库任务

在既有 Codex 配置中，用 `task_factory` 替换 `reward_callable`，两者必须且只能设置一个。
以下是已实现的接口片段，不是经过 NPU 验收的完整训练 recipe：

```yaml
agentic:
  runner: codex
  codex:
    task_factory: examples.code_agent.task:build_task
    task_config:
      run_timeout: 5
    workspace:
      image: sha256:REPLACE_WITH_BUILT_IMAGE_ID
      max_concurrent: 1
      output_limit_bytes: 4194304
    max_request_bytes: 8388608
    max_response_bytes: 33554432
```

任务工厂接收 prompt 与受信 `task_config`，返回异步 `prepare(workspace)` / `evaluate(archive)` 对象。
候选镜像和 grader 镜像保持一致，`task_config.image` 默认继承 `workspace.image`。
repository 路径拒绝宿主 `workspace_template`、MCP 服务及自行配置的候选网络。
既有文本奖励路径保留。DeepSeek 仓库路径同样选择 `task_factory`，沿用停止后导出和独立评分；
候选内实际工具为 DS 的 `bash` 等工具，模型须在 `bash.command` 中执行编辑与验证命令。
DS SDK 在受信控制端管理 JSON-RPC，会把 runtime 固定启动在候选容器内；候选镜像需预装
0.1.1rc1 runtime、Cordis 配置、Python 3.12 relay 解释器及 Codex patch 命令。
模型通道仅接受固定 `/v1/chat/completions`，预算终止和传输失败的判定与 Codex 仓库路径一致。
`docker/Dockerfile.swebench-ds` 从本地固定 SDK 镜像和 SWE-bench 任务镜像离线构建派生镜像：

```bash
docker build --network none --pull=false \
  --build-arg TASK_IMAGE=sha256:REPLACE_WITH_SWEBENCH_IMAGE_ID \
  -f hyper_parallel/rl/docker/Dockerfile.swebench-ds \
  -t hyper-parallel/hyper-swebench-ds:local /tmp/empty-build-context
docker image inspect --format '{{.Id}}' hyper-parallel/hyper-swebench-ds:local
```

构建前创建空的 `/tmp/empty-build-context` 目录；新镜像 ID 必须同步写入受信 registry 和
`workspace.image`，并在该镜像上重跑原始/参考补丁官方基线。
仓库模型推荐再沿独立的 `Dockerfile.swebench-ds-bash`、
`Dockerfile.swebench-ds-bash-4k`、`Dockerfile.swebench-ds-bash-4k-compact`
依次构建固定派生镜像。对应 Cordis 配置分别关闭无关 `skill/job_*`，限制每条
stdout/stderr 入模字节，再将 DS token meter/压缩插件的窗口设为实际 16K。
每一层镜像 ID 变化后都需更新 registry 并重跑相同官方原始/参考对照；压缩会额外消耗
模型调用预算，且高熵工具内容仍可能在压缩前越界，HTTP 400 不能当作任务零分。
四卡完整配置见 [`qwen3_30b_a3b_swebench_deepseek.yaml`](configs/qwen3_30b_a3b_swebench_deepseek.yaml)；
它使用本地构建的完整镜像 ID，`/results/registry/registry.json` 必须是该镜像的受信 registry。

DS 示例接口（`workspace.image` 必须替换为本机离线构建的完整镜像 ID）：

```yaml
agentic:
  runner: deepseek
  deepseek:
    version: "0.1.1rc1"
    provider: deepseek-official
    model: policy
    session_root: /results/sessions
    gateway_port: 18880
    timeout_seconds: 1800
    request_timeout: 1800
    task_factory: examples.code_agent.swebench_task:build_task
    task_config:
      registry_path: /results/registry/registry.json
    workspace:
      image: sha256:REPLACE_WITH_DS_SWEBENCH_IMAGE_ID
      memory: 4g
      cpus: 2
      pids_limit: 256
      max_concurrent: 1
```

候选始终 `network=none`，容器内只监听 loopback 模型入口。控制端通过独立 `docker exec` 的 stdin/stdout
转发请求：Codex 固定 Responses 路由，DeepSeek 固定 Chat Completions 路由；上游目标和 session 凭据
由控制端指定，候选 URL/请求头不能改写它们。
不需要 bridge、宿主防火墙变更、候选端口发布或宿主 socket 挂载。
当前通道只支持本机 HTTP Gateway 与固定镜像内 Python 路径，不是通用网络代理。
响应完整且有界地读入后传给 CLI；本地 write/flush 成功不等于 CLI 已消费的 ACK。

Gateway 管理路由要求独立凭据，runtime 生成后在受信训练进程间同步，不写入 YAML、候选配置或轨迹。
`gateway_admin_host` 可单独指定管理访问地址；不设置时沿用 `gateway_public_host`。
直接构造 repository program 时必须显式传入 `admin_url/admin_token`。
注册超时/取消仍清理；幂等删除并保留已释放 session 标记，拒绝迟到注册复活。

候选槽位由同一 `session_root/.workspace-slots` 下的进程间文件锁控制，覆盖 candidate 到 grader 全周期；
同节点 owner 必须使用相同 session root 和容量。进程退出会释放锁，残留容器仍按 run ID 清理。
`workspace.run_id` 可指定整次实验身份，默认使用 episode session ID。

字节预算同样约束终态快照，快照包含全部调用，可能比单次响应大；超限明确拒绝，不裁剪原始动作、概率或上下文。
CLI 输出预算由 `workspace.output_limit_bytes` 控制；`request_timeout` 控制请求/通道及槽位等待，
`timeout_seconds` 控制 CLI 总执行时间。传输失败保留 Gateway 已有采样证据、标记不可训练，不能变为模型零分。
仓库任务的隔离 CLI 配置另设 `tool_output_token_limit = 10000`，覆盖固定 CLI 对未知模型的
10,000 字节工具历史回退上限；这不改变容器总输出预算。仍应分段读取大文件，避免反复整文件读取撑满模型上下文。
仓库任务必需显式设置 `agentic.codex.model_context_window`（CLI 窗口 C）；它可小于
`rollout.vllm.max_model_len`（服务上限 W）。配置校验要求 `0 < C <= W`，且
`floor(0.9 * C) + rollout.max_new_tokens + 512 <= W`，为 CLI 约 90% 的自动压缩阈值预留
一次模型输出和固定余量。固定 CLI 对未知模型使用 272K 回退窗口，不能用来管理较小的实际上下文。
这是必要的保守配置检查；实际提示长度、tokenization 和压缩时机仍可能使请求超过服务容量。
运行时 vLLM HTTP 400 仍归为基础设施故障且不可训练，不能改判为普通模型零分。
仓库任务关闭无关 skills 和交互工具；模型侧仅保留可执行的 `exec_command`、`write_stdin`，
并显示候选容器真实的网络、权限与路径约束。Gateway 保留未改写的 CLI 原始请求作为审计证据。
固定 CLI 自动压缩时会另发无工具摘要请求；Gateway 通过原始 `request_kind` 校验，
摘要也计入调用预算并保留采样。截断或空摘要为不可训练失败，不以摘要后的错误历史继续运行。
编辑可在候选容器内执行 `python /opt/hyper-codex-home/checked_patch.py <<'PATCH'`，后接完整
`*** Begin Patch` 到 `*** End Patch` 块及 `PATCH` 结束行。该命令打印编辑前后 hash 和有限 diff；
若 CLI 返回零但没有文件变化，或改动的 Python 文件存在语法错误，则返回非零。语法错误时已修改的
文件仍保留，并报告路径及行号供后续修复；读回相关源码并运行聚焦测试后再结束任务。
`exec_command` 没有 `stdin` 参数；补丁必须作为 shell heredoc 写在 `cmd` 字符串内。Gateway 会在执行前
拒绝模型生成的未声明工具参数，保留原始采样并把具体错误反馈给模型重试，不能将额外字段静默丢弃。
已归因模型格式失败按 M3 零分规则结束，不用残留代码补救评分。

仓库普通模型调用预算到期时，Gateway 仅在完整记录且没有其他失败时返回结构化终态。
控制端排空通道、主动停止容器并冻结已有代码，再运行相同独立 grader；奖励可为 0 或 1，不预设为零。
全部调用保留并共享 episode 奖励，轨迹标记 `truncated=true`、`terminal_reason=max_completions`，
原始模型结束原因和 token/logprob 不变。固定 CLI 的精确预算重试诊断只在已确认主动停止的上下文中接受，
其他 CLI/传输/停止异常仍拒绝更新；文本奖励和格式重采样失败合同不变。
真实 Docker/CLI 的预算正例、负例及自然结束验证证明接线与独立评分，不代表真实模型修复或 NPU 训练通过。

## 验证入口

在依赖齐备的控制环境中，从仓库根目录 editable 安装并核对导入来源：

```bash
python -m pip install --no-deps --no-build-isolation -e . -e hyper_parallel/rl
python -c 'import hyper_parallel, rl; print(hyper_parallel.__file__); print(rl.__file__)'
export PYTHONPATH="$PWD/hyper_parallel/rl/tests/st:$PWD/hyper_parallel/rl/tests/trial:$PWD/hyper_parallel/rl:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q \
  hyper_parallel/rl/tests/trial/test_docker_workspace.py \
  hyper_parallel/rl/tests/trial/test_repository_task.py
```

`PYTHONPATH` 为当前尚未打包的 `examples` 提供源根，不代替 editable 安装。
统一镜像中既有 RL 导入链可能加载 NPU 运行库；CPU 控制容器需提供运行库，
可用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 关闭 Torch 自动加载，但这不消除所有显式导入。
无需映射 NPU 设备。候选/grader 执行纯 Python 仓库，不依赖这条 RL 导入链。

显式运行真实 Docker 验收，`--image` 替换为构建所得完整 ID，输出目录必须为新目录：

```bash
python hyper_parallel/rl/tests/trial/_repository_baseline.py \
  --image sha256:REPLACE_WITH_BUILT_IMAGE_ID \
  --output hyper_parallel/rl/output/code-agent/stage-a/run-unique \
  --run-id code-agent-a-unique
```

脚本验证两个缺陷版本得 0、参考修复各得 1，以及真实文件增改删、公开测试、
停止后快照、CLI patch helper、目录写权限、资源隔离和 6 类失败路径。
只有全套断言及所属容器残留扫描通过后才写 `completed.json`。
参考修复由控制端显式写入，用于验证环境；这不是模型自主完成的代码修改。
原始 tar、manifest、parquet 和报告写入忽略的 `output/`，不提交生成产物。

准备后续接线使用的数据：

```bash
python -m examples.code_agent.prepare_data \
  --output hyper_parallel/rl/output/code-agent/tasks.parquet
```

阶段 B 新合同位于 `tests/trial/test_repository_config.py`、`test_repository_program.py`、
`test_gateway_transport.py`、`test_model_relay.py`；共享 Gateway/runtime 修改应一并回归既有 RL CPU 测试。
真实容器接线实验使用相同控制环境和镜像：

```bash
python hyper_parallel/rl/tests/trial/_repository_program.py \
  --image sha256:REPLACE_WITH_BUILT_IMAGE_ID \
  --output hyper_parallel/rl/output/code-agent/stage-b/run-unique \
  --run-id code-agent-b-unique
```

要显式检查固定 CLI 的长文件输出能否进入下一次请求，给上述命令加 `--long-output-check`；
要检查真实候选编辑命令、反馈和非空冻结补丁，改加 `--checked-patch-check`。
要检查未声明的 `stdin` 参数在执行前被拒绝、纠错反馈进入下一轮且合法 heredoc 能编辑，
改加 `--stdin-schema-check`。
要检查自动压缩的无工具摘要请求及随后恢复的普通调用，改加 `--compaction-check`。
各模式使用不同的新输出目录和 run ID，均不调用真实模型或 NPU；编辑模式故意产生错误修复，
独立 grader 应给 0。相关 CPU 合同还包括 `tests/trial/test_code_agent_edit_helper.py`。

该实验运行真实 Codex CLI、Gateway、容器工具与 grader，但上游返回固定测试动作和合成 token/logprob。
它只验证调用接线、奖励/episode 归属和原样映射，不证明真实模型 token 一致性、自主修复或 RL 学习效果。

## 真实训练功能验收

已验证的入口为 [Qwen3-4B 配方](configs/qwen3_4b_code_agent.yaml)：16 次调用预算、16K 上下文、2K 输出，
使用精简的可选 Codex 系统提示并关闭 thinking；默认 CLI 提示不受影响。
在受信训练控制环境中，按实际挂载修改配方的模型、数据、输出路径及空闲端口，再显式运行：

```bash
torchrun --standalone --nproc_per_node=2 \
  hyper_parallel/rl/tests/trial/_repository_train.py \
  hyper_parallel/rl/examples/code_agent/configs/qwen3_4b_code_agent.yaml \
  /results/acceptance --acceptance functional --require-uneven-calls
```

`functional` 检查两步 optimizer、实际 token 对齐、版本发布、零损失 DP 补齐、最终评估及 checkpoint，
允许自然全同奖励并记录零优势/梯度/参数差值。省略 `--acceptance` 默认采用更严格的 `learning` 门禁，
还要求自然混合奖励和非零任务更新，不能用参考补丁或额外奖励满足。
对应逻辑测试为 `tests/trial/test_repository_training.py`。
训练与评估复用两个功能仓库；checkpoint 与 Hugging Face 导出已验证写出。
同模型、同两卡拓扑和 constant LR 已验证从 step 2 恢复后继续一个 step：optimizer 30→38、采样 policy 2→发布 3，
完成评估与 step 3 保存；不保证与未中断训练位级一致或跨拓扑恢复。
恢复验收入口为 `tests/trial/_repository_resume.py CONFIG OUTPUT`（使用同样两卡 torchrun）：
将 `train.checkpoint.load_path` 指向只读源 step 2、`train.max_steps=3`，输出与 session 路径另设。
本次恢复冒烟将调用预算统一设为 8，主配方仍为 16；恢复评估为 0/2，正常退出码 0。
该专属 trial 会在恢复前污染一个 live 参数以验证真正还原，源文件不改；禁止把此检查直接嵌入生产训练。
[MoE 配方](configs/qwen3_30b_a3b_code_agent.yaml) 尚未通过仓库训练验收，不作为已验证运行入口。

## 清理与边界

正常路径由 `close()` 回收容器；超时或取消先停止整个工作区，避免工具子进程继续运行。
控制端突然退出时，使用本次唯一 run ID 定向列出遗留容器，人工检查后只清理列表中的本次资源：

```bash
docker ps -a --filter label=hyper-rl.owner=code-agent \
  --filter label=hyper-rl.run-id=code-agent-a-unique
```

不可执行全局 prune 或清理其他实验。daemon 延迟完成创建时需要再次扫描，
不能把控制端 finally 当作突然退出时的自动回收保证。
当前所有候选/grader 均 `network=none`；模型请求只经受控通道，管理凭据不交给候选。
当前只支持固定小型 Python 仓库与普通文件，不支持任意仓库、链接、二进制产物或在线安装依赖。

## SWE-bench 小规模接入

`examples.code_agent.swebench_task:build_task` 使用固定 SWE-bench v4.1.0 TestSpec 和官方 `resolved` 评分。
首批仅支持开发文档固定的两个 pytest 7.2 实例，ARM64 Python 3.9、Codex 0.152.1；
不是完整 SWE-bench 环境构建器，不支持任意仓库、语言或任务集。
固定镜像按官方选例 TestSpec 安装依赖，不保证整个仓库的可选测试依赖齐备；例如全库收集可能因
缺少 `hypothesis` 或 `xmlschema` 失败。此类候选工具反馈不代表测试逻辑失败，应与独立官方评分分开记录。
扩展测试依赖须构建并固定新镜像、重验原始/参考补丁基线，不能原地改变 registry 对应环境。

控制端 `registry.json` 保存原始任务、参考/测试补丁、官方源码与 eval 脚本 hash、公开基线和每实例完整镜像 ID。
`swebench_data.py` 仅将公开 issue 与 registry 身份投影为训练 parquet；所有私有字段保留在控制端。
受信任务的 `workspace_image` 可选择该实例镜像，candidate/grader 必须使用同一完整 ID；
快照沿用工作区默认的 16 MiB/4096 成员预算，当前两例公开源码约 4.68 MB。

候选可修改 `src/_pytest/` 和 `src/pytest/` 内普通 UTF-8 Python 源码，不能改测试、配置、执行位或
`src/_pytest/_version.py`。控制端用冻结字节对固定基线生成 Git patch，不读取候选 Git 索引；
`.git`、字节码和 egg-info 不进入重放。每个 grader 使用新的无网络容器，严格应用 patch 并执行完整官方脚本。
官方必需 F2P/P2P 状态齐全时才使用 `resolved` 二元奖励；命令退出 0 本身不代表通过。
同一 Python 3.9 中先确认基线语法合法，再对候选变更执行 compile（不执行源码）；
明确语法错误为可训练零分，记录编译证据且不伪造官方测试报告。
其他收集失败、状态缺失、超时和未知错误仍拒绝该组。

`docker/Dockerfile.swebench` 在本地官方实例环境上增加独立 CLI/bridge 与断网 editable 所需 wheel。
官方 x86 Conda 构建号的 ARM64 适配、实际镜像及环境 hash 见开发文档；不能将可变镜像标签当作任务身份。
新增 CPU 合同在 `tests/trial/test_swebench_{artifacts,task,data,training}.py`，
真实基线入口为 `tests/trial/_swebench_baseline.py`，真实 rollout 复用 `_repository_rollout.py`；
`configs/qwen3_4b_swebench.yaml` 与 `_swebench_train.py` 为阶段 D 两步功能验收配方和观察器。
两例 train/eval 重叠，最终报告仅证明功能，不代表完整 benchmark 或泛化能力。

当前两卡功能配方使用 6 次模型调用预算、32K 推理容量。预算到期正常冻结，并保留全部已采样内容。
32K 服务容量不代表两卡可训练任意 32K 轨迹：当前全词表 FP32 概率计算仍有显存峰值限制，
长轨迹曾明确 OOM，实际可用长度须按硬件和运行证据判断。

在已安装固定官方评估器、当前 checkout editable 和镜像的受信控制容器中，
每次将新的结果目录挂到 `/results`，先按开发文档准备并验收其中的 `registry/registry.json`，再运行：

```bash
python -m examples.code_agent.swebench_data \
  --registry /results/registry/registry.json --output /results/tasks.parquet
python -m torch.distributed.run --nproc_per_node=2 \
  hyper_parallel/rl/tests/trial/_swebench_train.py \
  hyper_parallel/rl/examples/code_agent/configs/qwen3_4b_swebench.yaml /results/acceptance
```

启动前仍须选择两张健康空闲卡并保留镜像 CANN 环境。此命令显式运行 trial 验收观察器，
不加入默认 UT/ST 门禁；`completed.json` 只在实际训练、发布、评估和保存要求均满足后写入。
