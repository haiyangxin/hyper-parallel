# 仓库级 Code Agent 续接交接

更新：2026-10-09。当前开发分支为 `master-code-agent`，本轮整理内容随该分支提交。
本文维护当前验收状态与下一步；实现和复现入口见[开发说明](code_agent_development.md)与
[仓库示例](../examples/code_agent/README.md)。历史日志、轨迹、实验检查点和临时观察器
已按用户要求清理，保留实现代码和必要说明；下表记录已完成实验，不表示清理后重新运行过。

本轮整合后使用上述固定运行镜像完成 CPU 回归：`tests/ut/rl/`、
`tests/ut/auto_models/models/qwen3_5/` 和 `tests/ut/core/fully_shard/test_param.py`
合计 604 项通过。共享配方测试夹具已去重，原断言保留；本轮未重跑 NPU 实验。

## 当前验收状态

| 模型与范围 | 结果 | 边界 |
| --- | --- | --- |
| Qwen3-4B 训推一致性 | TP1/TP2 两步正式 bit-exact 通过，22530 个动作 token | 学习验证按用户安排留后 |
| Qwen3.8-27B 基础 RL | GSM8K 四步 GRPO、851 参数完整发布、保存及新容器恢复通过 | 有真实混合奖励、非零梯度及参数变化；不代表 Code Agent 训练通过 |
| Qwen3.8 Code Agent 推理 | 两个可控任务和两个固定 SWE 实例修复、独立评分通过 | 外部 API 与 native-vLLM 分别验收；原生四例共 55 次调用、90 项测试 |
| Qwen3.8 Code Agent 最小训练 | 一次 Actor 更新、851 参数发布及新版本短推理通过 | 复用完整原生轨迹；未包含完整在线两步、最终评估与保存 |
| Qwen3.8 Code Agent 完整训练 | 未通过 | 最后一次八个可控任务均奖励 1，但第一步反向发生 NPU OOM；已按用户要求停止重试 |
| Qwen3.5-9B 可控任务 RL | 两步训练、427 参数发布、最终评估和 DCP/HF 保存通过 | 16 个训练会话及四个评估奖励全零，未观察到修复或学习 |
| Qwen3.5-9B 恢复续训 | 新容器精确恢复并完成步骤 2→3，优化器计数 32→48 | 模型、优化器、调度器、CPU/NPU RNG 与数据游标均恢复；原累计 token 字段不作为流量统计 |
| Qwen3.5-9B SWE RL | 两步功能闭环通过，最长实际输入加输出 14349 token | 训练 16 例、评估 4 例均奖励 0；目标测试 20 次失败、既有回归测试 780 次通过 |
| Qwen3.5-9B BF16 数值对照 | 原门限未通过 | rank 0 的近零 `A_log` 局部梯度相对 L2 为 0.0625，超过 0.05；绝对差约 2.24e-8 |
| Qwen3.5-9B 更长对话容量 | 未完成 | 两次探测分别中断和 NPU OOM，均未形成完整会话，不计分或训练 |

Qwen3.5 的功能通过不替代上述数值门禁，不宣称该模型 bit-exact。
Qwen3.8 的最后一次完整 Code Agent 实验在 `FlashAttentionScoreGrad` 发生显存分配失败，
Actor 更新、后续发布、第二步、评估及保存未完成。本次精简没有重跑 27B，不能认定其阻塞已解除。

## 实际预算与运行环境

| 设置 | Qwen3.5-9B 正式配方 | 已完成的功能实验 |
| --- | --- | --- |
| 每会话模型调用 | 100 次 | 8 次 |
| 每次输出 | 12288 token | 相同 |
| Codex / Native 上下文 | 69632 / 81920 token | 相同 |
| Thinking | `reasoning_effort: none`，实际模板 `enable_thinking=false` | 相同 |
| 训练 / 评估温度 | 1.0 / 0.7 | 相同 |
| 每提示采样 | 8 | 2 |
| CLI 总时限 | 5400 秒 | 相同 |

8 次预算用于优先验收流程，不能代替正式预算下的模型能力评估。24 次探测未完成，
其中最长 18448 token 仅为中断会话的请求长度，不是已通过的训练容量。
Qwen3.8 已通过的原生推理沿用外部 API 的 128K、默认 thinking 和不限调用次数设置；
仓库中的小预算 27B YAML 是开发入口，两者不可混为同一实验。

当前验证镜像为 `hyper-parallel/hyper-rl:qwen3_8-v0.23.0.post1-swe4.1.0-arm64`，
完整 ID 为 `sha256:821524123dbbb88990c22fd15bedff5e62487c105f13a48ae58612859096d169`。
关键版本为 Torch 2.10.0+cpu、torch-npu 2.10.0.post4、Transformers 5.5.4、
vLLM 0.23.0+empty、vLLM-Ascend 0.23.0.post1、CANN 9.1、SWE-bench 4.1.0。
原始模型位于 `/home/xhy/Project-hw/models`；构建源码和 wheel 位于
`/home/xhy/Project-hw/reference`。安装与镜像说明见[Docker 文档](../docker/README.md)。

## 下一步与复现

1. 先检查当前 editable 导入、模型身份、空闲卡、磁盘、端口和候选镜像。不要复用已删除的旧产物路径。
2. 重建 SWE registry、原始失败及参考补丁成功对照；数据使用 `--repeats 2` 支持四卡功能采样。
3. 若继续 9B 长对话验证，先用独占单卡和 24 次预算取得两个完整同提示样本。
   单卡自动 KV 预算曾 OOM；待验证的资源修订为 `kv_cache_memory_bytes=8589934592`，
   保留 81920 上下文及其余设置，启动后核对实际缓存容量和真实 worker 版本 0。
4. 只用完整、同源原生 token/logprob 轨迹验证实际更长输入的 Actor 容量；中断或外部 API 记录不可替代。
5. 有效学习需真实混合奖励、非零优势、梯度及参数变化，功能通过或自然全同奖励不能替代。

正常最多四卡，必要时可到六卡；八卡须先获用户同意。连续三次被 kill 后拆分实验，
OOM 与外部退出分别记录，不能将失败补成零奖励。Docker 或源码访问需要代理时使用宿主机
8990；模型下载按用户要求显式禁用代理并优先国内源。不要操作其他任务或全局 prune。

### DeepSeek Harness 对照：工具、输出与下一步动作

已有 DeepSeek 仓库接线与隔离工具合同；其模型对照结果不替代 Codex 或当前模型验收。
使用独立配方和固定候选镜像，保留既有失败传播、输出预算和独立评分合同。

### 外部 API 仅推理验收（2026-09-28）

Qwen3.8 外部 API 曾完成两个可控仓库与两个固定 SWE 实例修复。
此入口返回 `inference_only` 结果，不提供训练策略版本或原生逐 token 概率，禁止输入训练。
用法和鉴权边界见[外部 API 入口](../examples/code_agent/README.md#外部-api-仅推理验收)。

## 7. 下一阶段：Qwen3.8 RL 的最小接入范围

基础 RL 已完成，待解决的是 Code Agent 完整长轨迹训练的显存与最终验收。
复用 `models/qwen3_5/`、现有 GRPO 和流式 full_gather，不新增训练后端。
详细边界见[模型接入说明](qwen3_8_development.md#2026-10-08-qwen38-完整模型与-rl-验证)。
