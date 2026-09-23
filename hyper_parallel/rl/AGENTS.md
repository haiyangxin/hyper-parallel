# 当前 RL / Code Agent 本地开发约定

本文件随本地开发提交保存；准备正式 PR 时再整理本机环境约定。先读仓库根 `AGENTS.md`、`.agent/rules/hyper-rl.md`、`docs/code_agent_handoff.md` 和 `docs/code_agent_development.md`。

## 开发基线与范围

- 当前续接分支为 `master-code-agent`，从 `rl-migration-pr` 的 `431fe12d` 创建，并迁入原 `master-dev` 中 M3 之后的 Code Agent 功能。`master-dev` 只作旧实现参考；`rl-migration-pr` 正在独立审核，勿为 Code Agent 开发改写或推送。
- 保留当前 RL 包和文件职责，新增文件按相邻目录组织。若确需重组现有结构，先说明路径、原因和调用方影响并取得用户同意。
- 改动 `hyper_parallel/rl/` 之外的 HyperParallel 公共运行代码前，说明为何 RL 内无法解决、影响哪些公共调用方，并取得用户同意。RL 导航和 UT 按仓库规则维护。
- 复用已有真实逐调用 prompt/action、episode GRPO、零损失 DP 补齐和失败归因。服务故障、断连及未知失败不得变成正常低奖励；仓库 grader 要对冻结 artifact 独立评分。
- 先定位当前空补丁轨迹中的 CLI 截断、工具/提示与未真正编辑问题，再决定是否增加推理预算或重跑模型矩阵。功能正确性优先。外部 API Qwen3.8 已完成两个固定 SWE-bench 实例的推理修复验收，不能据此宣称训练接入或有效学习。下一步按交接文档验证 Qwen3.8 训练，首版限定 FSDP、训练 TP=CP=1 与 full_gather 权重同步。

## 验证与资源

- 正式 CPU UT 在 `tests/ut/rl/`，真实 ST 在 `hyper_parallel/rl/tests/st/`；Code Agent A–D 的研究合同和观察器暂在 `hyper_parallel/rl/tests/trial/`，显式运行。之后准备 PR 时按普通 UT/ST 的既有组织方式整理研究测试，不能把需镜像/NPU 的测试混进普通 UT。
- 权重在 `/home/xhy/Project-hw/models`，数据在 `/home/xhy/Project-hw/data`，训练镜像为 `hyper-parallel/hyper-rl:v0.22.1rc1-unified-arm64`。运行前确认当前 checkout 的 editable 导入和实际镜像 ID，不复用过期 worktree/site-packages。
- 正常最多四张卡，有必要可用至六张卡；八张卡必须事先征求用户同意。连续三次实验被 kill 后拆分实验并保留原验收断言；中断不能算通过。
- 网络先正常访问，失败后试 8990 代理，再失败则显式禁用所有代理直连；保留真实错误，不输出凭证。候选和 grader 容器无网络、无 NPU、无 Docker socket；受信控制容器的历史特权配置仅用于已授权的对应实验。
- 历史证据在忽略目录 `hyper_parallel/rl/output/code-agent/`，不随提交。每次新分支验收必须区分当前运行和旧快照；测试通过后精简并记录实际证据、未运行项和边界。
