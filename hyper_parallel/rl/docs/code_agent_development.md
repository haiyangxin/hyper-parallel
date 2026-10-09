# 仓库级 Code Agent 开发方案与进展

更新：2026-10-09。当前实现以 Codex CLI、隔离仓库、独立评分和逐调用 GRPO 为主线。
本文保留实现边界、验收结论与复现入口；当前状态统一见[交接文档](code_agent_handoff.md)。
历史逐轮日志、失败重试列表和实验生成权重已按用户要求删除。

## 目标与支持边界

模型自主查看真实仓库、修改源码及运行测试，系统冻结提交并独立判题，
再用完整原生调用轨迹计算 GRPO、更新 Actor、发布策略并继续采样。
当前验收只覆盖两个可控 Python 仓库及两个固定 SWE-bench 实例，不代表完整 benchmark 或泛化收益。

首版使用同步 GRPO、FSDP、训练 TP=CP=1、共卡 native-vLLM、consistency off、eager 和 full_gather。
Qwen3.5-9B 与 Qwen3.8-27B 使用现有 `qwen3_5` 公共 AutoModel 和模型目录中的适配器。
不另建 Ray、VERL、Kubernetes 或新的 Trainer；保留已有 DeepSeek、单轮 code、PPO 和 MoE 的独立合同。
Qwen3-4B 的一致性路径仍优先，不能用新模型功能结果替代其 bit-exact 验收。

## 实现归属

| 边界 | 文件 | 职责 |
| --- | --- | --- |
| 模型构建 | `models/qwen3_5/adapter/`；`rl/roles/model_setup.py` | 公共文本模型、完整官方参数身份及模型输出能力 |
| 分块概率 | `models/qwen3_5/adapter/selected_log_probs.py` | 512 行 head 投影与重算，返回完整 FP32 下一 token 概率 |
| RL 配置 | `rl/config.py`；`examples/code_agent/configs/` | 已支持模型范围、原生 parser、预算和公开配方 |
| 外部程序 | `rl/agentic/codex/{harness,gateway,protocol}.py` | 固定 CLI、请求转换、真实逐调用证据与失败传播 |
| 仓库隔离 | `rl/agentic/envs/{docker_workspace,model_relay}.py` | 候选生命周期和受控模型通道 |
| 独立评分 | `examples/code_agent/{task,swebench_task,swebench_artifacts}.py` | 冻结字节、允许路径、可信基线及原判题 |
| 训练与同步 | `rl/trainer.py`；`rl/roles/policy/actor.py`；`rl/roles/weight_sync/` | 原 Reference/GRPO/Actor、完整版本发布 |
| 保存恢复 | `rl/checkpoint.py`；`core/fully_shard/hsdp_param.py` | 正确 HF 身份、DCP 恢复和无旧 autograd 图的分片重建 |

以上路径相对于 `hyper_parallel/rl/`，`models/` 与 `core/` 相对于 `hyper_parallel/`。
必要的共享 HSDP 修复保留其主项目测试，不把公共模块规则移入 RL。

## 生命周期与安全边界

```text
SyncTrainer → ProgramAgentRunner → 固定任务与策略身份
  → 隔离 candidate / 固定 Codex CLI
  → stdio relay → controller Gateway → native-vLLM
  → 完整请求、prompt/action、raw logprob、工具反馈
  → 停止 candidate → 固定 tar / manifest / hash
  → 全新 grader → 一次任务奖励
  → 完整 episode GRPO → DP 零损失补齐 → Actor
  → 全参数流式发布 → 版本提交 → 下一轮采样
```

candidate/grader 无网络、无 NPU、无宿主源码、权重和 Docker socket，使用固定镜像和资源限制。
仅受信控制端持有 Docker 能力、管理凭据和模型上游鉴权；凭据不进入候选或训练记录。
共享槽位覆盖候选到 grader 全周期，不能每 rank 各开一套独立无限并发。
详细协议由[Agent 合同](agentic_rl.md)和[仓库示例](../examples/code_agent/README.md)维护。

## 数据、评分与训练语义

- 可控仓库只允许 `src/*.py`，每份提交执行全部五项私测，全部通过才奖励 1。
- SWE 只使用固定官方版本的 FAIL_TO_PASS 与 PASS_TO_PASS，原始代码失败、参考补丁成功先验收。
  本轮固定 `pytest-dev__pytest-10051`（16 项）和 `pytest-dev__pytest-10081`（64 项）。
- 候选停止后从可信基线和冻结字节生成补丁，不相信候选的 git 索引、测试入口或成功声明。
- 每次调用使用真实 `P_i + A_i`，只对其动作部分计算损失；不 decode/re-encode 重建 token。
- 每个完整 episode 只计算一次奖励及 GRPO 优势，再映射到全部调用；DP padding 不计任务或有效动作。
- 基础设施故障、断连、未知失败拒绝更新；已证实的错误模型动作或合法错误代码才按任务合同零分。
- 调用额度耗尽时，仅在完整终态、无其他失败且完成冻结判题的情况下接受真实奖励，不预设为零。
- 外部 API 入口是 `inference_only`，不能制造原生 token、old logprob 或 Actor 策略身份。

分块输出仍计算所有 prompt、动作和 padding 位置的下一 token 概率，损失掩码不变。
普通 HF forward 和参数身份保留；root 必须 `reshard_after_forward=false`，head 不独立分片，
避免不同 rank 的分块次数产生不同数量的集体通信。该能力位于模型目录，RL 只做必要接线。

## 验证与复现入口

从仓库根目录安装当前 editable 并核对导入，参考[运行环境说明](../docker/README.md)。
普通 CPU 测试在 `tests/ut/rl/` 与模型 UT；真实模型和分布式验证使用显式 ST。
研究脚本位于 `hyper_parallel/rl/tests/trial/`，不会自动代表普通 PR 门禁通过。

```bash
export PYTHONPATH="$PWD/hyper_parallel/rl:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m examples.code_agent.prepare_data --output /results/tasks.parquet --repeats 2
python -m examples.code_agent.swebench_data --registry /results/registry/registry.json \
  --output /results/swebench.parquet --repeats 2
```

SWE registry 必须重新由 `_swebench_baseline.py --instances ... --images ... --output ... --run-id ...`
生成并验证原始/参考对照；实例数据和镜像身份须对应固定官方版本，不能用删除前的 hash 冒充新资产。
正式配方的挂载路径、镜像 ID、模型、端口和输出目录需与实际控制环境一致。

```bash
torchrun --standalone --nproc_per_node=4 \
  hyper_parallel/rl/tests/trial/_repository_train.py CONFIG /results/acceptance \
  --acceptance functional --require-uneven-calls

torchrun --standalone --nproc_per_node=4 \
  hyper_parallel/rl/tests/trial/_swebench_train.py CONFIG /results/swe-acceptance
```

每次使用新的输出目录。可控任务省略 `--acceptance` 默认执行更严格的学习门禁；
SWE 使用原官方评分，不套用可控仓库五项测试。保存、恢复、修复和学习分别报告。
恢复入口 `_repository_resume.py` 要求同模型/拓扑，源 checkpoint 只读，目标输出另设。
清理后不存在旧 checkpoint，须先生成新的成功保存后才能恢复。

### Qwen3.8 原生 Code Agent 联调（2026-10-08）

基础 GSM8K 四步 RL、完整发布和新容器恢复已通过，见[模型验收](qwen3_8_development.md#2026-10-08-qwen38-完整模型与-rl-验证)。
Code Agent 最小闭环也通过：两个完整原生 episode、12 次调用、3354 个动作，
原 Reference/GRPO/Actor 三次优化器步骤、851 参数/339 桶发布及版本 1 短推理。
同奖励造成零优势，不能据此声称任务学习。

#### 复用已通过的外部 API 配置

Qwen3.8 原生推理沿用此前 API 的 Codex、原提示和工具、128K 上下文、默认 thinking、
不限模型调用次数及服务默认生成参数。HF BF16 与外部 API Q6_K 不作数值等价声明。
四例均修复并通过独立评分：word_counts 5 次调用、merge_intervals 6 次，
两个 SWE 实例各 22 次；共 55 次调用、42794 个输出 token 和 90 项原测试。
独立推理允许图执行；RL 发布仍使用已验证的 eager 路径。
仓库中的 8K/1024 输出小预算 27B YAML 属于开发配方，不能替代这组原生推理设置。

### 2026-10-09：最后一次完整实验，失败后停止

最后一次 27B 完整在线 Code Agent 实验的八个新可控任务均奖励 1：
61 次调用、16642 个动作完成真实评分和 Reference。
第一步 `FlashAttentionScoreGrad` 发生 `207001 / EL0004` NPU 显存分配失败，
Actor 未完整返回，后续发布、第二步、最终评估和保存未完成。
按用户要求停止重试。本次代码精简未重跑 27B，不能认定该阻塞已解除。

## Qwen3.5-9B 接入与剩余验收（2026-10-09）

保留 Codex，参考本机 Agent Lightning 的部署与训练设置：正式 100 次调用、单次输出 12288、
Codex/服务窗口 69632/81920、thinking off、训练/评估温度 1/0.7、LR 1e-6、
clip 0.2/0.28、CPU offload、完整激活重算、推理 TP1/DP4。
与 Mini-SWE-Agent 不同的提示、工具、逐调用损失和输出预算保留当前仓库合同。

已完成功能实验使用每会话 8 次调用、每提示两个样本，正式配方仍为 100 次、每提示八样本。
可控任务两步验收通过，随后新容器恢复步骤 2→3、优化器计数 32→48，评估及保存通过。
SWE 两步也通过：113 次训练调用、10621 个动作、最长实际输入加输出 14349 token，
427 参数/130 桶两次完整发布、策略 0→1→2、四例评估及 DCP/HF 保存，自然退出 0，约 48 分钟。
训练 16 例和评估 4 例均奖励 0，20 次目标测试失败、780 次已有回归测试通过。
功能闭环通过，有效学习和自主修复未观察到。

CPU 完整概率、梯度和重算检查通过。BF16/HSDP4 原数值门限仍失败：
rank 0 `A_log` 局部梯度相对 L2 0.0625 超过 0.05，绝对差约 2.24e-8；不放宽或称 bit-exact。
此前一轮中断试验曾有真实混合奖励及非零更新，它不替代完整运行的学习验收。

24 次预算长对话探测尚未完成：第一次服务退出 137，来源未知；同配置重试发生 NPU OOM。
两次均未形成完整会话，不计分或训练。待验证的单卡配置仅新增 8 GiB KV 缓存预算，
保留上下文及其余参数，启动须核对实际缓存容量和 worker 版本。历史临时脚本已清理，
后续验证从上述正式代码入口重新组织，不依赖已删除的轨迹、registry 或 checkpoint。
