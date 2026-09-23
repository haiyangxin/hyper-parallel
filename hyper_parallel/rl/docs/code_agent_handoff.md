# 仓库级 Code Agent 续接交接

更新：2026-09-29。本文件给接手的 AI 一个从当前开发分支开始的入口。详细设计、阶段实验与证据在[开发记录](code_agent_development.md)；运行配方在[仓库示例](../examples/code_agent/README.md)。迁入前的历史实验来自原 `master-dev`；下文单列本分支的新验证，不能把旧结果当成新分支验收。

## 接手速览：当前状态与下一步

- **已完成：**当前仓库 Code Agent 经 Qwen3.8 外部 API 完成两个可控任务和两个固定 SWE-bench 实例的真实修复、冻结及独立评分。下方历史空补丁记录描述当时的 Qwen3 试验，不代表当前 Qwen3.8 推理验收失败。
- **未完成：**Qwen3.8 的本地 vLLM-Ascend rollout、HF Actor 训练、权重发布、GRPO 更新和恢复。外部 API 路径是 `inference_only`，不可直接用于训练。
- **最新决策：**用户希望下一阶段接入 Qwen3.8 RL；首版限定 FSDP、训练端 TP=1/CP=1、`full_gather` 同步，同步期间暂停 rollout。暂不扩展 `direct_reshard` 或复杂并行组合。
- **本地提交：**迁移基线为 `431fe12d`，Code Agent 工作合并为其后的一个提交；原 `74e110a5` 将被 amend 替换，实际新 SHA 用 `git log -2 --oneline` 查询。仅本地提交，不推送、不修改 `rl-migration-pr`。
- **先做什么：**阅读第 7 节并核对现有模型构建与同步代码，确认 HF 权重、运行镜像及资源，提出具体实施方案后按 RL 规则一次确认。当前批准的整理提交不表示 Qwen3.8 训练已经实现或验收。

## 1. 先确认基线

- 唯一工作仓库：`/home/xhy/Project-hw/hyper-parallel`；当前续接分支：`master-code-agent`。该分支从 `rl-migration-pr` 的 `431fe12d` 创建，移入原 `master-dev` 在 M3 `b76d18b6` 之后的 Code Agent 提交 `d4621556`。实际 HEAD 和工作区状态以 `git status -sb`、`git log -2 --oneline` 为准。
- `rl-migration-pr` 是 GitHub PR #958 的单提交审查分支，已通过 GitCode !1652 的静态检查、ARM/X86 编译、CPU UT/ST、Ascend ST 和 coverage。不要为了 Code Agent 开发修改或推送该分支。`master-dev` 留作迁移来源快照；旧交接中的“继续在 master-dev 开发”已过期。
- 首先读根 [AGENTS.md](../../../AGENTS.md)、[RL 规则](../../../.agent/rules/hyper-rl.md)、[Agent 合同](agentic_rl.md)与[功能边界](moe_code_agent.md)。当前仓库为原生 Torch。涉及 RL 目录外公共运行代码时，先解释原因和调用方影响，取得用户同意。
- 本分支只在本地建立，不是已提交的 Code Agent PR；不要将历史 A–D 功能、最新诊断与 PR #958 的 M1–M3 验收混为一谈。

## 2. 已迁入的能力

| 层 | 入口 | 已实现的行为 |
| --- | --- | --- |
| 受控仓库任务 | `examples/code_agent/task.py`、`fixtures/`、`tasks.json` | 两个可控 Python 仓库；候选读文件、修改、公开测试，冻结 artifact 后独立 grader 给真实奖励 |
| 工作区隔离 | `rl/agentic/envs/docker_workspace.py`、`docker/Dockerfile.code-agent` | 控制端管理容器，candidate/grader 无网络、Docker socket、权重和 NPU；限制资源与输出，校验归档路径和内容 |
| 模型与控制通道 | `envs/model_relay.py`、`codex/gateway.py`、`ds_harness/gateway.py` | 固定 session 模型 relay；管理接口独立鉴权；有界请求/响应、断连与失败传播 |
| 训练闭环 | `codex/harness.py`、`core/program_runner.py`、`rl/config.py`、`checkpoint.py` | 复用 M3 真实逐调用 P/A、episode GRPO 和 DP 零损失补齐；可控任务两步更新、权重发布、同模型同拓扑恢复续训 |
| SWE-bench | `examples/code_agent/swebench_{data,artifacts,task}.py`、`docker/Dockerfile.swebench` | 固定两例、基线与参考补丁对照、真实产物和官方评分；两卡短训练只验功能 |
| 研究入口 | `hyper_parallel/rl/tests/trial/` | CPU 合同、真实仓库/训练观察器和 SWE-bench 基线脚本；是显式运行的开发试验，不属于 PR #958 的 UT/ST 门禁 |

保留原有单轮 Python stdio/SandboxFusion 路径；仓库级任务使用独立 Docker 判题。候选自然耗尽正常调用预算时冻结现有文件交给真实 grader；断连、超时、未知故障仍拒绝训练。最终答案文本或模型自报成功不能代替文件变更或官方分数。没有接入相同前缀调用的拼接优化，也没有分段 PPO。

## 3. 历史验证及准确边界

- A：两个可控仓库的真实读改测、独立判题及隔离失败路径曾通过。B：Codex 容器接线、管理鉴权、relay 与传输故障合同曾通过。
- C：Qwen3-4B 两卡真实模型完成两步训练、策略 0→1→2、参数更新与保存；另做同模型同拓扑 policy 2→3 的恢复续训。功能闭环不证明任务学习、成功修复或跨拓扑恢复。Qwen3-30B 配方存在，但仓库任务训练未作对应真机验收。
- D：SWE-bench Verified revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`，官方 evaluator v4.1.0 / `726c5461e2ef52d83cf1ea2107870a8bb3328d57`；固定 `pytest-dev__pytest-10051`、`pytest-dev__pytest-10081`。原始代码失败、参考补丁通过，两例两卡短训练及独立判题的功能路径曾通过；模型补丁均为空，官方评分 0，不代表 SWE-bench 已解决或有效学习。完整 benchmark 不要求运行。
- 暂停前的扩展诊断只完成第一例四组：thinking off/12 和 off/24 用尽调用预算；on/12 和 on/24 均在第六次自然结束。四组都为空补丁，官方 0。第二例新矩阵尚未完成，不能合并旧试验结果宣称全矩阵结论。
- **迁入阶段的 CPU 验证：**RL 全量 CPU UT `514 passed`；15 个 Code Agent 研究合同文件 `153 passed`；受影响的 M3/rollout 定点回归 `105 passed`（与全量 UT 重叠，不能相加）。研究 worker 直接导入、所有改动 Python 语法、Markdown、AGENTS catalog 和新增相对链接检查通过。迁入时尚未重跑真实 Docker 工作区、模型/NPU 训练、官方 grader 或恢复试验；后续新分支真机结果在下节单独记录，旧日志不能替代它们。三项新增研究测试若依赖 Docker/权重/模型，不要混入普通 UT。

## 4. 空补丁目前定位到哪里

完整轨迹审查已排除“冻结/导出通道总是丢失修改”：此前有真实保存的非空错误补丁。失败更靠前：

1. thinking off：读取 30010 字符整文件后，实际下一轮 prompt 只保留头尾各约 5000 字符，关键 `reset/get_records/clear` 没进入模型上下文。随后无效 grep 循环，没有发出写文件命令。截断来源已定位为固定 CLI 对未知模型的 10000 字节回退上限；工具说明的 10000 tokens 不能覆盖它。
2. thinking on：未先读源码，先向不存在的 session 1 写入拟议 diff，接着出现 sed 语法错误、JSON 非法转义以及 sed 未命中却以零状态退出；未复查文件就宣称修复。两组均未耗尽预算。
3. 模型实际可见短错误反馈；命令与真实 shell 脚本匹配；已修复旧 Hermes reasoning 中示例 `<tool_call>` 被误算为实际工具失败的问题，并保留严格证据校验。尚未证实工具/提示配置与空补丁的因果关系。
4. 候选侧全库测试收集缺少 `xmlschema/hypothesis` 等可选依赖，但官方指定测试完整运行。环境收集失败不能冒充官方未解决或模型成功。

下一步按单因素顺序：**确认 Codex CLI 实际输出截断与模型收到的 prompt → 整理真实工具、权限、网络及编辑后 diff 检查 → 再比较有限推理预算与模型能力**。记录每轮工具命令、真实 stdout/stderr、冻结前后文件 hash 和官方评分。不要直接扩大矩阵或以生成更多 token 掩盖工具问题。Actor 全词表 FP32 `log_softmax` 与最长序列 padding 可能造成显存峰值，但本轮未实现或验收优化；不能未经梯度等价与峰值测量就宣称修好。

续接分支对仓库任务设置 CLI 的 `tool_output_token_limit = 10000`。
固定 0.152.1 CLI 的无模型双轮回归确认一次 30010 字符文件输出完整进入下一次原始请求，中段可见；
冻结空变更经独立 grader 判 0。既有可控仓库合成读改测仍判 1。
证据在忽略目录 `output/code-agent/cli-limit-20260923-2/` 和 `cli-limit-regression-20260923-1/`。
它们不是模型/NPU 或多次长读取的上下文验收；CLI 未知模型回退为 272K 上下文，实际 vLLM 配方为
16K/32K。经用户批准，仓库任务已关闭无关 skills/工具，将模型侧权限说明校正为候选 Docker 约束，
移除模型可见的不可用审批参数；CLI 原始请求保留原文。候选新增 `checked_patch.py`，编辑后输出真实
文件 hash/diff，零状态无变化返回失败。固定 CLI/Docker 的合成编辑试验确认下一次原始请求收到验证反馈、
冻结补丁非空且独立 grader 仍能判 0；参考修复仍判 1。证据在忽略目录
`output/code-agent/tool-contract-{long-output,checked-patch,repair}-20260923-1/`；相关 trial CPU 合同 56 项通过。
独立 review 后加了仓库 MCP 配置拒绝和 Gateway 工具集合校验，trial CPU 合同 58 项通过；
固定 CLI/Docker 编辑回归复验在 `output/code-agent/tool-contract-reviewed-20260923-1/`。
仓库配置必须设置与 vLLM 上限一致的 `model_context_window`，纠正未知模型 272K 回退；压缩触发后的
实际请求和工具历史须从真机轨迹核对。本分支的新轨迹如下。

本分支四卡 30B 单例 `pytest-dev__pytest-10051` 已真实完成 12 次调用：prompt 峰值 15923，
两次完整 30010 字节源码读取的中段进入下一次模型请求，短工具错误也可见；模型反复用正则搜索
含方括号的字面源码，未调用编辑 helper 或运行测试，最终冻结空 patch，官方 16 项得 0。
初次控制容器缺固定 evaluator 的预检失败没有模型调用，不计入上述结果；复测使用 hash 与 registry
吻合的旧受信控制容器，证据在忽略目录 `output/code-agent/stage-d/branch-rollout-20260923-1204z/`。
示例提示已加入字面搜索与失败后换方法的通用规则；后续结果仍须与首轮分开记录。
第二次同例 12 调用复测在 `output/code-agent/stage-d/branch-rollout-20260923-1227z/`：模型
使用 `rg -n -F` 修正正则误用，但仍三次重复整文件读取、未编辑或测试，空 patch、官方 0；
prompt 峰值 22135，无压缩或上下文错误。提示改善工具语法的观察不等于修复成功。
24 次调用的下一组诊断在第 19 次普通调用后遇到 CLI 无工具摘要请求，被旧双工具校验误拒绝；
Gateway 将其记录为不可训练基础设施失败，未冻结/评分，不可计为模型零分。现按固定 CLI
`request_kind` 严格区分普通调用与压缩，后者无工具、摘要上限 1024 token，长度截断/空摘要拒绝，
成功压缩的固定提示不再误判致命。固定 CLI/Docker 的三调用压缩回归在忽略目录
`output/code-agent/compaction-contract-20260923-2/` 通过，原失败证据在 `...-1/`；
相关 CPU 合同 84 项通过。中断前 19 次不能作为最终补丁验收；修复后的复测如下。
修复后的 24 调用真机单例在 `output/code-agent/stage-d/branch-rollout-20260923-1250z/`
完整评分：24 次均为普通调用，prompt 峰值 18490，无压缩/上下文错误；没有编辑或测试，
空 patch、官方 0。增加预算仍只观察到分段读取和搜索，真机压缩路径尚未触发。
最后一次 12 调用同例复测加入完整 `checked_patch.py` shell 模板及六次探索上限；
首次服务先被 SIGKILL（exit 137）导致 0 completion 的基础设施失败，不记零分。相同四卡服务
重启后用新运行身份成功跑完，第六次已定位 `self.handler.reset()`，仍无编辑/测试，空 patch、
官方 16 项得 0，prompt 峰值 15897。配置、失败和重试证据分别在忽略目录
`output/code-agent/stage-d/branch-rollout-20260923-1259z/`、
`output/code-agent/stage-d/branch-rollout-20260923-1303z/`。工具模板入模但模型未执行；
不要把此结果写成可用 patch 或 RL 学习成功。
该批次新建服务/控制容器已移除、2–5 卡无残留进程；旧诊断容器未动，临时 `auth.json` 已删除。
对 1250z 的完整 24 次轨迹再次逐轮核对：19 次搜索、5 次读取，零编辑和零测试。第 3 次读取已让
`LogCaptureHandler.reset`、`LogCaptureFixture.get_records/clear` 的相关源码进入第 4 次实际模型 prompt；
第 7 次开始重复先前成功的搜索。23 条有后续调用的工具反馈逐字进入实际 prompt，24 次命令参数
只有 `cmd/login` 且真实执行对应，因此这条空补丁不能归因于编辑参数被丢弃。证据为
`output/code-agent/stage-d/branch-rollout-20260923-1250z/` 和
`output/code-agent/stage-d/branch-rollout-20260923-token-visibility-audit.json`。

为单独验证编辑能力，新分支又用四卡 30B 跑了小型 `word_counts` 真实任务。修复前的
`output/code-agent/word-counts-edit-20260924-0702z/` 中，模型第 1 次读完三个公开文件，随后 9 次
都试图调用 `checked_patch.py`，却把补丁放入 `exec_command` 未声明的 `stdin` 参数；固定 CLI
只执行 `cmd` 并忽略该字段，helper 九次收到空输入并退出 2。第 3 次起补丁块格式本身已正确，
但错误反馈只说补丁不完整，没有指出输入未传递。冻结文件未变化，独立五案评分 0。
这证实一个具体工具参数适配缺陷，但不能反向解释没有编辑尝试的 SWE-bench 1250z 轨迹。

Gateway 现仅对仓库任务在验证 parser/token 原始证据后按模型实际看到的 CLI 工具 schema
校验调用参数；未声明字段在执行前拒绝并给出 `cmd` heredoc 提示，原始动作和 token 仍完整记录。
工具描述也明确补丁应放在 `cmd` 中。相关 trial/agentic CPU 回归 81 项通过。
固定 CLI/Docker 的三调用合成回归在 `output/code-agent/stdin-schema-20260924-1/` 通过：
非法 `stdin` 调用在 CLI 执行前被拒绝，精确反馈进入下一次模型请求，随后合法 heredoc
真实修改候选文件并进入冻结 manifest；五案独立 grader 对故意未修任务仍判 0。
修复后相同小任务在 `output/code-agent/word-counts-edit-20260924-0716z/` 第 2 次调用即通过
`cmd` heredoc 完成真实源码编辑，helper 返回修改前后 hash 和 diff，第 3 次运行公开测试。
模型的补丁意外减少内层循环缩进，测试报 `IndentationError`，后续失败补丁未修正；
冻结 manifest 确认 `src/solution.py` 已修改，独立五案均为运行错误、评分 0。
固定 CLI 的 patch 引擎会在精确定位失败后尝试忽略前导空白匹配，这解释了错误缩进为何仍能写入。
候选 helper 因此增加写后 Python 编译反馈：语法错误返回非零并报告路径/行号，已修改文件保留以供继续修复；
独立 grader 不变。新增 trial 语法错误合同及原有 helper 合同共 7 项 CPU 测试通过。
在固定候选镜像的无网络 CPU 容器中重放模型原样坏缩进命令，helper 真实退出 1，报第 22 行
`IndentationError`，文件 hash 已改变且仍保留；证据为
`output/code-agent/word-counts-edit-20260924-0716z/syntax-smoke.json`。
该复测证明编辑与导出链路可用，不证明模型能稳定修复；思考模式单变量对照另行记录。
思考模式对照位于 `output/code-agent/word-counts-edit-20260924-0724z/`：相同任务、服务、
四卡、十次预算和评分，只将 `reasoning_effort` 改为 medium 并启用思考。模型十次都生成合法的
`cmd` heredoc，却未读源码，补丁始终引用不存在的 `def solve(lines): pass` 等旧行；helper
十次真实执行后均报告未命中，反馈进入后续模型调用。没有修改或测试，冻结 manifest 无变化，
独立五案评分 0。两次各一条样本只说明失败形态不同，不证明思考模式的总体效果。
修正工具合同后又以四卡 16K 服务对 `pytest-dev__pytest-10051` 做单例 12-call 复测，
证据在 `output/code-agent/stage-d/branch-rollout-20260924-0746z/`。模型分段读取并搜索到相关
`clear/reset/get_records` 源码，但普通调用仍全部只读，没有编辑或测试。第 11 次模型 completion
是固定 CLI 自动发出的无工具 `compaction`：420 个输出 token、正常 stop，完整摘要进入第 12 次
普通请求，后者仍执行搜索；摘要占用一次 12-call 预算。最终冻结 patch 0 字节，官方 16 项
F2P 失败 1、P2P 成功 15，评分 0，无基础设施故障。此结果不否定工具编辑通路的正例，
只表明该真实 SWE 轨迹即使看到源码与新工具描述，也未选择写入动作。
相同任务 24-call、16K/4096 输出预留的对照在 13 次只读普通调用后收到 vLLM HTTP 400：
服务要求输入加预留输出不超过 16384，而固定 CLI 的约 90% 自动压缩阈值尚未触发。
Gateway 正确记录为不可训练的基础设施失败；该运行未冻结、未官方评分，不能记模型 0。
证据在 `output/code-agent/stage-d/branch-rollout-20260924-0755z/`。
仓库 Codex 配置现于启动时检查 CLI 窗口 C、服务窗口 W 和普通输出预留 R：
`0 < C <= W` 且 `floor(0.9*C)+R+512 <= W`；它能拒绝该已知不匹配，但不能保证
所有实际 prompt 都不超限，运行时超限仍按基础设施故障传播。两个 16K/2048 示例将 C 下调至
14336 以提前压缩，保留原有输出额度；32K/2048 示例保持原值。
把同一 24-call 诊断的普通输出预留改为 1024 后，四卡服务曾在新请求前以 exit 137 退出；
Docker `OOMKilled=false`，原因未证实，故该启动不计模型结果。一次受控重启后，
`output/code-agent/stage-d/branch-rollout-20260924-0801z/` 的单例正常完成：24 次 completion
含 1 次无工具压缩，压缩前后的相关源码分段均完整进入模型请求；23 次普通调用全为有效读/搜索，
没有调用编辑 helper 或运行测试，`def clear` 搜索重复 9 次。输出预留 1024 未再触发 HTTP 400。
冻结 patch 0 字节、base/content hash 相同，官方 16 项 F2P 失败 1、P2P 成功 15，评分 0。
完整逐轮审计在同目录 `model-trace-audit.json`。这条轨迹把剩余卡点定位于模型选择动作：
它获得了源码和反馈，却未发出写入动作；不能据此宣称模型在其他任务上无法编辑。
随后仅把同例非思考采样组合改成 Qwen 模型卡建议的 `temperature=0.7`、`top_p=0.8`、
`top_k=20`，保留权重、seed、提示、16K/1024、24-call 和 grader。
`output/code-agent/stage-d/branch-rollout-20260924-0827z/` 正常完成 24 次普通调用，
其中 20 次为相同且成功的 `def clear` 搜索，未编辑或测试；冻结空 patch、官方 16 项评分 0。
同目录 `fixed-prompt-diagnostic/audit.json` 记录无候选执行的单步采样：语义相同的模型请求
在基线/推荐采样和三个 seed 下，七次有六次继续搜索 `def clear`，一次搜索 `get_records`。
诊断请求因 JSON 字段排序使工具定义处的实际 prompt token 与原 rollout 相差九个，
因此它只说明这个停滞状态下的动作偏好，不能冒充位级相同的重放或完整 episode 验收。
本地权重是通用 `Qwen3-30B-A3B`，约 3.3B 参数每 token 激活；
[Qwen3 技术报告](https://arxiv.org/pdf/2505.09388)将其列在强到弱蒸馏路径，
未披露这份精确权重直接接受仓库级 Code Agent RL。该单例不能证明权重总体能力上限，
但修正工具、上下文和采样组合后仍无写入，当前 Code Agent 任务成功验收仍未达成。
推荐采样主 rollout 与单步诊断产物均已保存，之后模型服务 exit 137；
该退出不改写已完成的官方评分。新建控制容器已移除，临时 `auth.json` 已清理；
旧受信诊断容器保持原状。

### DeepSeek Harness 对照：工具、输出与下一步动作

`master-code-agent` 已将 DeepSeek Harness 0.1.1rc1 接到相同仓库任务工厂、隔离 Docker 候选、
控制端固定 Chat Completions relay、冻结归档与独立 grader。SDK 在受信控制端，真实 runtime
通过 `docker exec` 运行于无网络候选中；模型工具是 `bash`，不是 Codex 的 `exec_command`。
文本奖励路径保留。新派生镜像均固定完整 ID，官方 `pytest-dev__pytest-10051` 原始/参考补丁
对照在每个镜像上分别为 0/1，F2P 1 项、P2P 15 项均按预期。

CPU 真实 SDK、Gateway、Docker 和独立 grader 的可控 `word_counts` 回归得到：脚本化有效编辑
奖励 1，不编辑且公开测试失败奖励 0；一轮预算后冻结已修改文件仍奖励 1，调用轨迹标记
`max_completions`。再用 DS `bash` 真正运行候选 `checked_patch.py` heredoc：4 文件变更的
hash/diff 和公开测试通过反馈进入下一轮实际模型请求，grader 奖励 1。
因此 DS 编辑命令、模型工具调用格式、反馈和冻结评分链路已有正例，但脚本化模型不证明真实
30B 会选择编辑。对应证据见 `output/code-agent/stage-d/ds-program-{pos,neg,budget}-*/` 与
`ds-checked-patch-cpu-20260924-1/`。

四卡 Qwen3-30B-A3B 在同一个 SWE-bench 实例、16K 上下文、1024 输出预留、24-call 上限及
`temperature=0.7/top_p=0.8/top_k=20` 下的单例对照：

| DS 候选工具配置 | 真实轨迹结果 | 证据 |
| --- | --- | --- |
| stock：`bash`、`job_*`、`skill`，64 KiB 单流尾截断 | 首次 `bash` 广泛读取所有源文件，反馈 64,104 字符；第 2 次请求 HTTP 400。无编辑/评分，属于基础设施失败。 | `ds-rollout-20260924-0955z/` |
| 仅 `bash`，默认 64 KiB 输出 | 首次定位 `logging.py`，后三次错误组合 `tail -n +START -n 120`，三次返回同一 25,824 字节/707 行；第 5 次请求 HTTP 400。 | `ds-bash-rollout-20260924-1011z/` |
| 仅 `bash`，单流 4 KiB | 过限延后到第 15 次请求；目标源码被尾截断遮蔽，未编辑。 | `ds-bash-4k-rollout-20260924-1016z/` |
| 再明确 `sed -n 'START,ENDp'` 分段与重复输出处理 | 正常走满 24 次；第 5 次模型上下文已含 `clear/get_records`，仍只有检索，0 编辑/0 测试；冻结空补丁，官方 16 项评分 0。 | `ds-bash-4k-prompt-20260924-1022z/` |
| 再挂载 DS 原生 token meter/compaction，固定 16K 窗口 | 第 13 次摘要调用将下一次普通提示从约 8.9K 降到 3.0K tokens；24 次调用含 1 次摘要，全部采样 token/logprob 对齐，无 HTTP 400。摘要已指出 `clear`，模型仍只检索；空补丁、官方评分 0。 | `ds-native-compact-rollout-20260924-1033z/` |

各失败的 `model-trace-audit.json` 位于对应目录；Gateway 将 HTTP 400 和推理服务退出归为
不可训练基础设施故障，不能计作模型零分。第一次服务就绪后 worker 意外退出也只留下故障轨迹，
不算模型尝试。原生压缩 CPU 高熵合成输入暴露 token meter 可能低估并在摘要前超过 16K；
真实源码轨迹没有超限，不等于任意输出都安全。当前真实模型仍未完成 SWE 修复：
已证实它能看到目标函数且工具链可编辑，剩余空补丁主要是该任务上的下一步动作选择和
重复命令纠错问题。一个样本不能推出模型在其他仓库任务上的能力上限。

上述四卡轨迹在随后 Gateway 加固前运行；加固修复了预算 409 与无关 SDK 异常的区分、
模型可见闭合工具参数 schema、原始请求保真和后端拒绝请求的有界留证。
加固后的 CPU 定点回归及真实 Docker/SDK 脚本化编辑和预算正例已通过，
最终同例四卡复测第一次服务启动时 worker 在首请求前退出，准备受控重启时外部作业一度
占用 2–5 卡；两段预检均没有模型结果。资源释放后同配置受控重启完成真实 24-call 轨迹：
原 SDK 工具 schema 保持原样，模型实际看到 `bash.additionalProperties=false`，
第 5 次提示已含 `def clear/get_records`；后续重复 `caplog.get_records()` 搜索 3 次，
错误的 `find -exec ... \\;` 命令 9 次（3 次退出 1），0 编辑/0 测试；
结构化预算后冻结空补丁，官方 16 项评分 0，
无 HTTP 400 或其他基础设施错误。该轨迹提示只到约 9K tokens，未触发压缩；
压缩真实触发的验收由上表较早的四卡轨迹提供。预检、最终轨迹和评分证据在
`ds-final-rollout-20260924-1052z/`，不能把两次服务故障算作模型零分。

### 外部 API 仅推理验收（2026-09-28）

用户批准显式仅推理合同后，本分支接入 Qwen3.8-27B 外部 API，复用现有 Codex **0.152.1**、
Responses→Chat Gateway、双工具与 checked_patch、CandidateRelay、DockerWorkspace、任务准备、
停止后导出和原有独立 grader。入口为 `examples/code_agent/inference.py`，配置与边界见
[仅推理合同](../examples/code_agent/README.md#外部-api-仅推理验收)。没有用独立项目的新版 CLI
实验冒充本链验收，也没有扩大任务源码修改范围。

Gateway 与 program 必须同时显式选择仅推理；策略版本为 `None`，外部结构化响应与 usage 保留，
不构造训练 `Trajectory`，不伪造 token ID、logprob 或 Hermes 原始解析证据。默认训练入口保持严格，
即使推理记录碰巧含 token 字段也拒绝进入训练。API 密钥只在控制端使用，有限预算和无调用次数预算
均有独立合同检查；服务/协议/清理故障不能记作普通模型零分。

实际服务为 `qwen3.8-27b`、Q6_K、131072 窗口；保留服务采样默认，没有旧 24-call/1K 限制。
原始基线、公开任务、固定镜像和官方 evaluator 沿用受信 registry。四项均为一次真实模型尝试：

| 任务 | 模型调用 | 输入峰值 token | 独立评分 |
| --- | ---: | ---: | --- |
| word_counts | 14 | 7238 | 5/5，reward 1 |
| merge_intervals | 11 | 9176 | 5/5，reward 1 |
| pytest-dev__pytest-10051 | 22 | 11683 | F2P 1/1、P2P 15/15，resolved |
| pytest-dev__pytest-10081 | 14 | 9672 | F2P 1/1、P2P 63/63，resolved |

模型曾出现错误补丁或测试回归，真实失败反馈进入下一轮后自行修正。完整冻结产物、manifest 与评分
哈希已逐项核对，所有原始工具反馈在模型请求中可追溯。证据位于忽略目录
`output/code-agent/external-api-acceptance-20260928/`，总览为 `acceptance-summary.json`。
固定 CLI 的无训练 token 合成验收位于 `output/code-agent/inference-smoke-20260928-1/`：
真实异步 session 轮询、编辑增改删、公开测试及独立 5/5 评分通过。真实 Docker 四个正反例及六类
故障/清理检查通过；集中 CPU 合同结果与完整命令保存在总览引用的记录中，重叠回归不得相加。

这是当前工作区的**推理功能验收**，不是 Qwen3.8 训练、权重同步、GRPO 学习或整体 SWE 成功率验收。
独立项目先前的 Qwen3.8/Qwen3 对照使用 Codex 0.156.1 和不同上下文；它们仅作能力/归因参考，
不能混入上表。后续训练接入仍需 HF 权重和对应模型、训练/推理后端及采样证据的独立验证。

## 5. 环境、证据与接手命令

- 权重 `/home/xhy/Project-hw/models`，数据 `/home/xhy/Project-hw/data`。统一训练镜像 `hyper-parallel/hyper-rl:v0.22.1rc1-unified-arm64`，代码可用的控制容器和 SWE-bench 固定镜像身份见开发记录。实验正常最多四卡、确有需要可到六卡；八卡须先获用户同意。实验连续三次被 kill 后拆短，保留原验收断言。
- 历史证据在忽略目录 `hyper_parallel/rl/output/code-agent/`；D 的 registry、selection、官方 baseline、diagnosis-1、四组完整轨迹位于 `stage-d/` 下，逐轮审查是 `stage-d/diagnosis-1/full-rollout-audit-off.md` 和 `stage-d/diagnosis-1/full-rollout-audit-thinking.md`。它们留在本机，未进入 Git；丢失时不能补造结果。
- 从仓库根运行并确认 `hyper_parallel` 的 editable 安装指向当前 checkout。研究合同显式设源根和现有 ST worker 路径，例如：

```bash
export PYTHONPATH="$PWD/hyper_parallel/rl/tests/st:$PWD/hyper_parallel/rl/tests/trial:$PWD/hyper_parallel/rl:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q hyper_parallel/rl/tests/trial/test_docker_workspace.py \
  hyper_parallel/rl/tests/trial/test_model_relay.py \
  hyper_parallel/rl/tests/trial/test_repository_program.py \
  hyper_parallel/rl/tests/trial/test_swebench_task.py
```

其余显式试验见[仓库示例](../examples/code_agent/README.md)。先验新分支的 CPU 合同和可控仓库，再决定是否需要最小真机实验；不为“让门禁绿”跳过或削弱断言。若要将研究测试提交到后续 PR，再按现有 `tests/ut/rl/`、`hyper_parallel/rl/tests/st/` 目录职责归位并处理 CI 依赖。

## 6. 交付与分支纪律

- `master-code-agent` 承载后续 Code Agent 开发；`rl-migration-pr` 仍是已通过门禁、待审核的 M1/M2/M3 PR。两条分支分别验证和审查，不把 Code Agent 内容推到前者。
- 新分支沿用当前代码布局，不恢复已删除的 M1/M2/M3 迁移子文档，不把忽略的实验产物、权重或密钥送入提交。用户已要求取消 `AGENTS.md` 和 trial 的忽略，本次本地开发提交包含 RL `AGENTS.md` 与研究测试；正式 PR 前再按职责整理。代码优先修复真实瓶颈，阶段结束后精简并给出可复查证据。
- 后续 PR 按仓库规则保持一个审查提交。当前迁入的 A–D 工作先作为继续开发的完整基线；是否拆分 PR 及具体合入顺序要根据范围和门禁再决定，不能把历史真机结果当成新分支最新验收。

## 7. 下一阶段：Qwen3.8 RL 的最小接入范围

### 已有能力和真正缺口

不要从零重写 AutoModel、并行规划、FSDP 或 GRPO。公共 `HyperAutoModelForCausalLM` /
`HyperAutoModelForImageTextToText` 已负责 HF 构建、分片及加载；
`hyper_parallel/models/qwen3_5/adapter/registration.py` 已注册 `qwen3_5` / `qwen3_5_text`，
含 GDN 参数角色声明。其 CP wrapper 和测试已存在，但当前明确不支持 GDN 同时 TP>1、CP>1。
已核对的 Qwen3.8 模型配置使用该架构；适配存在不等于当前 Ascend 训练已通过。

| 边界 | 接手文件 | 要完成的工作 |
| --- | --- | --- |
| 模型识别与构建 | `rl/roles/model_setup.py`、`rl/config.py` | 目前 family 仅接受 qwen3/qwen3_moe；选择正确的公共 AutoModel，接入新 family 和配置校验 |
| 模型专用假设 | `rl/roles/qwen3_builder.py` | 不直接复用 Qwen3 Attention/RMSNorm 替换和固定权重绑定路径；先用公共 builder，仅补实际缺口 |
| 文本训练接口 | `rl/roles/policy/actor.py`、公共 AutoModel | 明确完整多模态模型的纯文本训练或文本子模型加载方式，验证权重名、视觉参数范围、logits、mask、梯度和 checkpoint 重计算 |
| 推理后端与容量 | `rl/roles/rollout/vllm.py`、`rl/config.py` | 选择并固定支持模型的 vLLM-Ascend/Transformers 组合；旧自动容量估算仅接受 full-attention Qwen3，不能直接用于 GDN 混合架构 |
| 权重同步 | `rl/roles/weight_sync/model_adapter.py` 及现有发布调用链 | 仅接 full_gather；核对名称、GDN 投影与融合布局，优先复用推理引擎原生 loader；不重写已有传输机制 |
| 训练轨迹 | `rl/agentic/codex/gateway.py`、`core/program_runner.py` | 恢复真实采样 token/logprob、逐调用 prompt/action 和策略版本证据，保持 inference-only 记录拒绝训练 |

full gather 省去跨训练/推理分片的直接重排，不自动解决模型参数命名、融合布局或推理状态清理。
核对现有实现是否逐参数/分桶聚合并及时释放；27B 的 BF16 参数约 54 GB，不能默认额外完整副本可驻留。
外部 API 的 Q6_K/GGUF 是推理对照资源，不能当作现有 HF Actor 的训练权重；先确认 HF checkpoint
是否完整、架构和 tokenizer 是否一致，再安排资源。GDN 在 Transformers/Ascend 的反向目前是待验证，
不能套用独立 llama.cpp 的 CPU 回退问题而宣称必需开发新算子。

### 按顺序验收

1. 同架构小配置：通过公共 builder 完成前向、反向、一次参数更新；验证 padding/序列隔离和重计算，比较分片前后数值。
2. 真实 HF 权重：在许可卡数内验证 FSDP 短序列训练、显存及保存恢复。正常四卡，必要时至六卡；八卡须用户批准。显存不够时说明实测瓶颈，不暗中扩大拓扑或改为另一训练方案。
3. full gather 发布：暂停 rollout，发布新版本并清理旧推理状态，核对训练参数与引擎加载结果、版本和更新后的推理输出；故障拒绝更新，不能自动回退。
4. Code Agent GRPO：先可控任务，再固定 SWE 实例；真实多轮采样、独立奖励、梯度更新、策略发布和恢复分别留证。任务修复成功与训练有效学习是不同结论。

### 当前证据与复现资源

- 本分支推理证据根：`output/code-agent/external-api-acceptance-20260928/`。读 `acceptance-summary.json`、`final-review.json` 和 `combined-contract-final-result.json`；最终集中 CPU 合同为单次 **129 passed、0 failed、0 skipped**，不是多个重叠回归之和。
- 服务在验收时为 `http://127.0.0.1:18000`、`qwen3.8-27b`、Q6_K、131072 上下文；服务器配置可能变化，重跑前查询。密钥文件在独立部署项目 `local-llm/configs/api-key`，只由控制端读取，不能输出或提交内容。
- 独立部署项目为 `/home/xhy/Project-hw/local-llm`；其 Codex 0.156.1 对照与本仓库固定 0.152.1 验收分开维护。独立 Qwen3-30B Q6/32K 单例仍为空补丁、官方 0，未发生窗口溢出；异步命令未正确轮询。该观察不证明模型在所有代码任务上的能力上限。
- 现有受信控制容器名为 `hp-code-agent-d-diagnosis-20260922`，代码挂载 `/workspace/hyper-parallel`；运行前检查容器、editable 路径和依赖，名称本身不是完整性证据。SWE 固定 evaluator、候选镜像与 hash 以 `output/code-agent/stage-d/registry/registry.json` 及验收总览为准。
- 模型/数据根分别为 `/home/xhy/Project-hw/models`、`/home/xhy/Project-hw/data`；旧训练镜像 `hyper-parallel/hyper-rl:v0.22.1rc1-unified-arm64` 不代表已经支持或验收 Qwen3.8。服务版本选择必须重新核对官方支持与实际安装版本。
- 最新验收已清理本轮候选、grader、session 和控制容器临时密钥；保留旧受信控制容器及外部 API。接手先检查实际状态，不关闭其他作业。

### 本地合并前检查（2026-09-29）

本轮仅整理文档及本地提交，未新增模型训练功能。合并前在上述控制容器显式运行
`python -m pytest -q -p no:cacheprovider hyper_parallel/rl/tests/trial`，源根设置沿用第 5 节：
**248 passed、0 failed，44.64 秒**，26 条 warning；不与此前 129 项重叠回归累计计数。
日志在忽略目录 `output/code-agent/pre-amend-20260929/trial-tests.log`。
本轮未重跑外部模型、NPU 训练或 SWE 真机验收，沿用第 4 节标明日期的推理证据。

合并整理期间补齐了 trial 辅助函数文档与一个类型标注，未改变执行逻辑。代码风格检查、
Python 语法、JSON、AGENTS catalog、9 份 Markdown 和 20 个新增相对链接检查通过。
AutoGit 整批 Pylint 超过脚本内置 120 秒后退出，完整 AutoGit check **未通过**；
此前推理验收的 Pylint 记录仍按原范围保留，不能当作本轮完整静态门禁。
本机未安装 pre-commit hook；本次是用户授权的本地开发快照，正式 PR 前须补完静态门禁及 trial 归位。
本轮检查日志与边界见同目录 `checks.json`、`autogit-check.log`。
