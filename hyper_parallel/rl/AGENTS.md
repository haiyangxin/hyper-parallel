# 当前 RL / Code Agent 本地开发约定

本文件随本地开发提交保存；准备正式 PR 时再整理本机环境约定。先读仓库根 `AGENTS.md`、`.agent/rules/hyper-rl.md`、`docs/code_agent_handoff.md` 和 `docs/code_agent_development.md`。

## 开发基线与范围

- 当前开发分支为 `master-code-agent`，主线基线为 `master`。按用户要求，完成本轮提交后清理旧开发和备份分支，本地与 `origin` 只保留这两个分支；后续开发从保留分支及其提交定位代码。
- 保留当前 RL 包和文件职责，新增文件按相邻目录组织。若确需重组现有结构，先说明路径、原因和调用方影响并取得用户同意。
- 改动 `hyper_parallel/rl/` 之外的 HyperParallel 公共运行代码前，说明为何 RL 内无法解决、影响哪些公共调用方，并取得用户同意。RL 导航和 UT 按仓库规则维护。
- 复用已有真实逐调用 prompt/action、episode GRPO、零损失 DP 补齐和失败归因。服务故障、断连及未知失败不得变成正常低奖励；仓库 grader 要对冻结 artifact 独立评分。
- 功能正确性优先，先定位模型轨迹、协议和容量问题，再决定是否增加预算或重跑。Qwen3.8 基础训练、native 推理和 GSM8K RL 已完成，完整 Code Agent 训练仍因 Actor backward OOM 未通过。外部 API 的固定 SWE-bench 推理修复不作为训练或有效学习证据。当前验收状态与下一步以交接文档为准。

## 验证与资源

- 正式 CPU UT 在 `tests/ut/rl/`，真实 ST 在 `hyper_parallel/rl/tests/st/`；Code Agent A–D 的研究合同和观察器暂在 `hyper_parallel/rl/tests/trial/`，显式运行。之后准备 PR 时按普通 UT/ST 的既有组织方式整理研究测试，不能把需镜像/NPU 的测试混进普通 UT。
- 权重在 `/home/xhy/Project-hw/models`，数据在 `/home/xhy/Project-hw/data`；运行时版本、镜像构建与安装以 [docker README](docker/README.md) 为准。运行前确认当前 checkout 的 editable 导入和实际镜像 ID，不复用过期 worktree/site-packages。
- 正常最多四张卡，有必要可用至六张卡；八张卡必须事先征求用户同意。连续三次实验被 kill 后拆分实验并保留原验收断言；中断不能算通过。
- 网络先正常访问，失败后试 8990 代理，再失败则显式禁用所有代理直连；保留真实错误，不输出凭证。候选和 grader 容器无网络、无 NPU、无 Docker socket；受信控制容器的历史特权配置仅用于已授权的对应实验。
- 按用户要求，长期只保留代码和必要文档；权重与数据输入保留。实验生成的运行目录、日志、提交归档、临时观察器/报告和 checkpoint 不保留为历史资产；清理前将实际结果、失败阶段和未验边界汇入现有交接文档。
- 后续实验需按代码和运行文档重建镜像、隔离工作区、判题资产及所需 checkpoint，并重新产生真实轨迹。不得依赖已清理的旧输出路径、把文档摘要当作原始验收证据，或把未完成实验补记为通过。
