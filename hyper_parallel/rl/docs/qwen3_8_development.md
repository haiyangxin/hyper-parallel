# Qwen3.8 RL 接入开发文档

更新：2026-10-09。Qwen3-4B bit-exact 回归和 Qwen3.8-27B 基础 RL 已通过；
Code Agent 推理及最小训练闭环通过，完整长轨迹训练仍受 NPU OOM 阻塞。
本文件保留模型和环境边界；当前 Code Agent 状态见[交接文档](code_agent_handoff.md)，
任务流程见[开发说明](code_agent_development.md)。历史日志和生成 checkpoint 已按用户要求清理。

## 模型与实现边界

本地 Qwen3.8-27B HF 配置声明 `Qwen3_5ForConditionalGeneration`、`qwen3_5`，
文本子配置为 `qwen3_5_text`，混合 GDN 与 full attention。名称与实际架构分别校验，
不能根据模型名称套用 Qwen3 dense 的 attention/RMSNorm 替换。

- 训练复用公共 `HyperAutoModelForCausalLM` 和 `models/qwen3_5/adapter/`。
- 初始 RL 范围为纯文本、GRPO、FSDP、训练 TP=CP=1、共卡 native-vLLM、consistency off。
- 原生服务使用 `--language-model-only`，显式 `max_num_seqs`，权重发布走 eager。
- 同步使用流式 full_gather 和共卡 IPC；覆盖全部官方文本参数后才提交新版本，失败不自动回退。
- 不引入新模型 PPO、direct_reshard、多模态 RL、异步或新的训练后端。
- Qwen3.5-9B 使用同一模型家族；其结果不能替代 27B 的内存和数值验收。

## 固定运行环境

| 库 | 已验证版本 |
| --- | --- |
| Torch | 2.10.0+cpu |
| torch-npu | 2.10.0.post4 |
| Transformers | 5.5.4 |
| vLLM | 0.23.0+empty |
| vLLM-Ascend | 0.23.0.post1 |
| CANN | 9.1 |
| SWE-bench evaluator | 4.1.0 |
| Codex CLI | 0.152.1 |

官方基镜像使用 `quay.io/ascend/vllm-ascend:v0.23.0.post1`，Agent 派生构建和
一致性依赖说明见[Docker 文档](../docker/README.md)。当前 SWE 派生镜像 ID 为
`sha256:821524123dbbb88990c22fd15bedff5e62487c105f13a48ae58612859096d169`。
标准镜像和严格一致性镜像各自核验依赖，不把仅版本号一致视为数值或模型支持通过。

源码和本地 wheel 保留在 `/home/xhy/Project-hw/reference`，原始模型保留在
`/home/xhy/Project-hw/models/Qwen3.8-27B`。配置 Docker 需要代理时使用宿主机 8990；
模型从国内源下载时显式清除代理并核对上游文件身份。运行前检查磁盘、镜像空间和 editable 路径。

## 模型、保存与发布接线

| 模块 | 必要实现 |
| --- | --- |
| `rl/roles/model_setup.py`、`rl/config.py` | 根据原 HF 身份选择公共文本 builder；限制已支持并行、算法和原生 parser |
| `models/qwen3_5/adapter/selected_log_probs.py` | 分块 head 概率与重算；普通 HF forward 不变 |
| `rl/roles/weight_sync/model_adapter.py` | 文本参数与原 conditional namespace 映射，不包含 vision/MTP |
| `rl/roles/weight_sync/vllm_worker.py` | 拒绝遗漏文本权重，保留流式桶确认、IPC 生命周期与版本提交 |
| `rl/checkpoint.py` | DCP 运行状态与匹配原模型身份的 HF 导出 |
| `rl/trainer.py`、`core/fully_shard/hsdp_param.py` | lazy CPU offload 初始化、角色释放、无 autograd 图的恢复分片 |

27B 有 851 个文本参数，BF16 发布量 53,791,996,928 字节，128 MiB 桶共 339 个。
全参数覆盖表示完整模型内容已发布，不表示先在一处聚集全部模型；沿原参数/桶流式处理，
确认和释放后再进入下一桶，最多一个在途桶。对应 9B 为 427 个文本参数和 130 桶。

新分块概率能力在 9B 完成功能验证后加入；当前未据此重跑 27B Code Agent，
不能宣称它已经解除此前 `FlashAttentionScoreGrad` 的显存阻塞。

### Qwen3-4B 正式 bit-exact 验收结果

TP1/TP2 各完成两步真实 RL，合计 22530 个动作 token，在开启一致性、LR 为零的条件下
通过正式 bit-exact 门禁。学习验证按用户安排留后；该结论不覆盖 Qwen3.5 或 Qwen3.8。
一致性依赖和已支持数值合同由[一致性文档](qwen3_training_inference_consistency.md)维护。

## 2026-10-08 Qwen3.8 完整模型与 RL 验证

完整 27B 四卡前向和反向通过。GSM8K 初始步骤 1/2、四样本评估、DCP/HF 保存及
重加载通过；新容器从步骤 2 恢复模型、优化器、scheduler、RNG 和数据游标后完成步骤 3/4。
四步覆盖 16 个 prompt、64 条真实响应、18948 个动作 token，均使用原判定器和真实采样。

各步有真实奖励差异、有限非零梯度和实际参数变化。AdamW 参数组计数 4→8→12→16；
恢复后数据正确续接，851 个私有 CPU 分片缓冲没有残留 autograd 图。
每次完整发布 851 参数/339 桶后提交对应 worker 版本；原初始评估四例正确三例。
恢复配置关闭最终保存，不能声称存在步骤 4 checkpoint。

HF 导出恢复外层 conditional 配置及原 namespace，Actor 仍是纯文本模型，
Reference 使用基础权重，不错误替换为训练后的 HF 导出。
本轮证明短程真实 RL 和恢复，不证明长期提升、训推 bit-exact 或 Code Agent 完整训练。

## Code Agent 已完成项与阻塞

两可控任务和两个固定 SWE 实例的外部 API、原生模型推理修复分别通过独立评分。
原生复现使用 128K、默认 thinking 和不限调用预算，共 55 次调用、90 项原测试。
API 的 Q6_K 与本地 HF BF16 不作数值等价声明；API 记录不能输入训练。

最小训练闭环通过：两个完整原生 episode、12 次调用、3354 个动作、一次完整 Actor 更新，
四 rank 各三次优化器步骤、851 参数/339 桶发布以及版本 1 的短推理。
两 episode 同为奖励 1，优势和梯度均为零，未证明 Code Agent 有效学习。

最后一次完整在线实验的八个任务均通过原五项私测，但第一步反向在
`FlashAttentionScoreGrad` 发生 `207001 / EL0004` NPU OOM。
Actor 更新未完整完成，后续发布、第二步、最终评估和保存均未完成；按用户要求停止重试。
SWE 长轨迹训练及 Code Agent 有效学习仍待独立验收。

## 后续验证原则

使用仓库保留的公共模型 ST、`_qwen3_5_train.py` 和 Code Agent 的原训练入口。
清理后先重建数据及 SWE 正反例 registry，不依赖已删除的临时观察器、日志或 checkpoint。
根据实际完整调用的长度安排内存验证，不能以 128K 服务声明推断训练容量。
原模型权重与正式构建 source/wheel 保留；中断不能评分或训练，显存不足不能暗中缩动作、
放松数值门限或扩大到八卡。Code Agent 详细预算与下一步见[当前交接](code_agent_handoff.md)。
