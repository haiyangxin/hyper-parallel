# 仓库级 Code Agent 开发方案与进展

日期：2026-09-22。状态：阶段 A/B/C/D 功能验收完成；SWE-bench 仅两例功能验证，未观察到成功修复或有效学习。
本文件维护本轮开发范围、接口、验收门禁和进展；已完成能力仍以[迁移功能说明](moe_code_agent.md)和
[Agent 合同](agentic_rl.md)为准。最新暂停状态、续接入口及分阶段 PR 说明见[当前交接](code_agent_handoff.md)。原始 `master-dev` 交接只作历史输入。

## 1. 目标与已确认范围

最终目标是：模型针对真实仓库问题，自主查看代码、修改文件、运行测试；系统冻结其提交，
在独立环境中评估，再将真实多轮调用及仓库级奖励用于 GRPO，完成更新、发布和后续采样。

用户已确认以下顺序：

1. 先以 **Codex + Qwen3-4B + 两个可控 Python 仓库**建立读改测、独立判题及训练闭环。
2. 再接入 **SWE-bench**，参考 slime 的真实 code agent 示例实现任务准备、补丁与评估流程。
   SWE-bench 是正式交付阶段，不以人工样例通过代替最终目标。

**硬件约束下以功能正确性验收：只运行资源可承受的少量代表性实例，不要求完整运行 SWE-bench，
不要求覆盖全部仓库、长期训练、达到指定解决率或证明 benchmark 分数提升。**
保留真实执行、评分、轨迹与训练合同检查；缩小实验规模不改变所选实例的判题标准。

首版采用同步 GRPO、共卡 native vLLM、consistency off；复用现有训练、权重同步和 episode 合同。
SandboxFusion 继续服务单轮 code；仓库执行与判题使用独立 Docker 容器。
不引入 E2B、Ray、另一套 Trainer/runner 或候选独立模型服务。DeepSeek Harness 仓库路径复用同一工作区、模型 relay 和独立评分合同。
前缀合并、候选级优化器调度、异步 rollout、Agent 子任务并行和全量并行策略矩阵不在本轮范围。

## 2. 基线与代码审视结论

唯一开发与验收仓库为 `/home/xhy/Project-hw/hyper-parallel`，当前续接分支为 `master-code-agent`，
基于 `rl-migration-pr` 的 `431fe12d`，叠加原 `master-dev` 的 Code Agent 实现 `d4621556`。
以下阶段日志记录当时基线和验收环境，不作为新分支已重跑真机测试的证明。
旧 fork 的冻结备份及 slime 均只读参考，不作为安装、挂载或运行时依赖。

| 当前实现 | 本轮处理 |
| --- | --- |
| `CodexAgentProgram` 在控制端目录执行 CLI，同步调用 `reward_callable(final_answer, prompt)` | 增加容器任务路径和异步产物评分，保留文本任务 |
| `ProgramAgentRunner`、逐调用 builder、`episode_rows()`、DP padding 已完成 M3 验收 | 直接复用，不重建 token、不增加另一套 episode 类型 |
| Codex 管理路由与模型路由共用地址，管理操作未单独鉴权 | 候选可达部署前补独立管理鉴权及地址接线 |
| Codex/DeepSeek SSE writer 吞掉 BrokenPipe/ConnectionReset | 保留采样证据并标记不可训练的传输失败；两条路径定点修复 |
| Gateway 请求和后端响应读取未统一限制字节数 | 增加有界读取、输出与并发限制；不能裁剪采样动作来过门禁 |
| Codex 注册请求在主要清理范围外 | 将注册尝试纳入生命周期；处理超时后服务端已注册的情况 |
| Actor 按调用行切 mini-batch | 保持现状，记录真实行数、padding 和 optimizer steps；不宣称 episode 等权 |

实现依据：[Codex program](../rl/agentic/codex/harness.py)、[Codex Gateway](../rl/agentic/codex/gateway.py)、
[共享 runtime/runner](../rl/agentic/core/program_runner.py)、[Actor](../rl/roles/policy/actor.py)。
上表记录开发前的审视与接入目标。B 已完成 program/Gateway 接线及失败修复；
训练优化语义保持不变，真实模型/NPU 训练仍须在 C 验收。

## 3. 组件与生命周期

```text
SyncTrainer → ProgramAgentRunner（仅 TP request owner 执行环境）
  → 绑定 task / episode / policy_version，申请本节点候选槽位
  → 准备公开基线与独立 candidate，注册 Gateway session
  → candidate 内 Codex CLI → 无网络 stdio 模型通道 → 共享 Gateway → 共享 native vLLM
  → 模型读文件 / 编辑 / 公开测试，Gateway 保存全部真实调用
  → CLI 收尾，封闭 session 并排空请求，停止 candidate
  → 导出、校验、固定 artifact 与内容 hash
  → 全新 grader 从可信基线重建，仅接收合法候选修改
  → 独立判题 → 一次 RewardResult → 现有逐调用 Trajectory
  → episode GRPO / DP padding / Actor 更新 / 权重发布
  → 清理本 episode 容器、session、临时资源
```

### 3.1 最小任务接口

拟增加 `agentic.codex.task_factory: module:function`，接收 `PromptRecord` 和受信任务配置，
返回具备 `async prepare(workspace)` 与 `async evaluate(artifact)` 的任务对象。
`evaluate` 返回已有 `RewardResult`；奖励、判题状态和产物身份随现有 metadata 记录。
工厂本身只验证配置和构造任务，不做阻塞 I/O。

- `task_factory` 与原 `reward_callable` 二选一；两者都缺失或同时设置，启动时拒绝。
- 通用 Docker 生命周期由 program/workspace 管理，任务仅负责基线、提交合法性和判题。
- 容器执行与本地文本执行共用 session、协议捕获、失败归因及轨迹构造；不复制整份 harness。
- 仓库路径在 candidate 镜像内部检查固定 Codex CLI 版本 `0.152.1` 和工具依赖。
  不能以控制端 CLI 版本代替容器版本验证，也不复制控制端登录凭据。
- Docker 调用及 grader 等待必须异步且可取消；取消后有界清理，不吞掉原始错误。
- 正常完成初版沿用 CLI 成功与最终答复合同；自然语言仅作为收尾信号，奖励只来自 artifact。
  非正常退出能否抢救已有修改不在初版扩展范围。

### 3.2 Docker 与通信

控制端持有 Docker 管理能力；candidate 和 grader 不挂载 Docker socket、训练源码、权重、数据根目录或 NPU，
不使用 privileged/host network，丢弃 capabilities，启用 no-new-privileges，限制 CPU、内存、进程数及命令输出。
训练容器若需管理宿主 Docker，只向受信控制端提供访问；部署时验证其可用性，不把 socket 传给子容器。

候选只获本 session 模型访问凭证。管理路由使用独立、每次运行生成的管理凭证；仅受信控制端持有，
不写入 YAML、候选配置、轨迹或普通日志。runtime 显式区分管理 URL 与候选可达的模型 URL。
鉴权先于管理操作和大请求体读取；候选用自身凭证调用注册、快照或删除接口必须失败。

最终采用更小的隔离方案：candidate/grader 都保持 `network=none`，不修改宿主防火墙或创建 bridge。
candidate 内 loopback HTTP 入口与控制端经独立 `docker exec` stdin/stdout 通信；控制端固定上游地址、
Responses 路由和 session 凭据，不转发候选提供的目标或鉴权头。CLI 看不到真实管理地址/凭据。
此通道在现有 `envs/` 新增 `model_relay.py`，不移动包结构；固定镜像的 stdlib Python 即可运行。
下游 HTTP 写失败也会回报 Gateway 基础设施失败，收束通道后才读取终态，避免把晚到故障遗漏为成功。

所有候选共用既有 rollout engine。候选槽位上限按本节点所有 request owner 合计，覆盖 candidate 到 grader 的完整生命周期；
不能每个 rank 各开一份不限总量的并发。首轮并发从 1 开始，功能稳定后验证 2。

### 3.3 冻结、提交与独立评分

控制端维护 task ID、基线内容身份、镜像完整 ID、测试版本和评分器版本。
candidate 仅接收公开问题与基线；参考修复、隐藏测试及预期结果留在控制端，隐藏测试仅在 grader 阶段注入。

CLI 收束后停止并确认 candidate 不再运行，随后导出只读产物。校验路径、重复条目、文件类型、总字节数及文件数，
拒绝路径越界、链接与特殊文件。传输超限与提交超限分别归因；导出失败不能伪装成模型零分。
评分使用已验证的冻结字节及其 hash，不重新读取可变工作目录。

人工样例先限定 `src/*.py` 的新增、修改、删除；其他基线文件受保护。
测试缓存、临时文件与 CLI 状态应定向放在提交目录外；只允许明确列出的非提交缓存被排除，
不能通过忽略规则隐藏受保护文件修改。记录三个变更集合，不只导出 tracked diff 而漏掉新增文件。

grader 每候选重新创建，从可信基线恢复保护文件并叠加合法修改，使用控制端固定入口。
人工样例由控制端比较实际输出与隐藏预期，不能采信 candidate 输出的 `passed=true` 或修改后的测试脚本。
一个候选给一次二元奖励；只有所有必需测试通过为 1，其余明确任务失败为 0。
Docker 是进程/文件系统隔离边界，不把共享内核容器宣称为虚拟机级安全保证。

## 4. 训练语义与失败分类

每次实际生成仍保存 `P_i + A_i`，仅本轮 `A_i` 参与动作损失，使用原始 token ID 与 L-1 对齐的 raw logprob。
即使前缀完全一致也不自动合并；不得 decode/re-encode 重建训练输入。
Codex 内部格式重采样也保留全部调用；工具观察进入下一次真实 prompt，但不作为本轮动作训练。

每个 episode 只计算一次仓库奖励和 GRPO 优势，再映射到全部调用行。
DP padding 不创建工作区、不重复评分、不增加有效 token；完整性及策略版本检查继续生效。
`response_mini_batch_size` 仍按调用行切片，可能拆开一个 episode；有效 token 均值损失不等于每候选等权，
一次 rollout 也不保证只执行一次 optimizer step。

| 情况 | 首版行为 |
| --- | --- |
| 正常完成、合法提交 | 运行独立 grader，按任务结果赋 0/1 |
| 错误代码、受保护路径修改、经确认的候选程序超时/输出超限 | 任务零分，保留合法完整的模型调用证据 |
| Gateway 已确认的模型格式错误及其重采样额度耗尽 | 沿用 M3 零分规则，不以遗留 artifact 覆盖失败 |
| 仓库普通调用额度耗尽且结构化终态完整、无其他失败 | 排空通道、控制端主动停止并冻结，独立 grader 真实赋 0/1；保留全部调用 |
| 文本 Codex 普通调用额度耗尽、未知 CLI 异常、原因不明的总超时 | 沿用拒绝更新规则；不能一律当模型零分 |
| Docker 启动/复制/停止失败、服务断连、后端错误、未知协议失败 | 基础设施/未知失败，拒绝整组更新 |
| SSE/JSON 交付失败 | 保存已有采样证据、标记不可训练、排空并清理；不当作格式错误重采样 |
| grader 缺依赖、测试安装失败、评分器解析失败 | 基础设施失败；不得把环境坏掉计算为候选失败 |
| 控制端取消或被 kill | 中断，无通过结论；按运行身份清理并按约定重试 |

判题执行超时只有确认属于候选代码且基线环境健康时才能零分；其他情况保守拒绝。
基础设施失败优先级高于后续模型失败。已 flush 不等于客户端消费 ACK，不承诺恰好一次交付。
注册请求超时仍以预先分配的 session ID 收尾；清理只能定位本 run/episode，不能广泛 prune。
控制端强制退出后由定向清理入口回收遗留容器，不依赖无法执行的 finally。

## 5. SWE-bench 接入设计

参考本地 slime 固定提交 `aaf5c2092b01219fa0d5c2d323741d409086ca32` 的
`examples/coding_agent_rl/generate.py` 与 `swe.py`，已阅读其真实代码。
重点借鉴 `prepare_workspace → git_diff → run_evaluation → _grade_swebench`，
复用 SWE-bench 的 `make_test_spec` 和 `get_eval_report` 语义。
不迁入 slime 的 Sample、E2B、SGLang/Ray、轨迹合并或缺依赖时默认给零分的处理。

### 5.1 数据、环境与评估合同

1. 任务清单固定数据集名称、revision、split、instance ID、repo、base commit、环境构建身份和测试版本。
   保留 `version`、`test_patch`、`FAIL_TO_PASS`、`PASS_TO_PASS` 等官方评估所需字段；仅公开问题进入 prompt。
   slime 的 `remote_env_info` 是其包装格式，可适配但不能误认为官方数据必有镜像字段。
2. 首批筛选 ARM64 可构建的小规模实例，分别验证原始基线失败、可信参考 patch 通过及既有测试保持通过。
   固定 SWE-bench 包/源码版本、依赖和镜像 digest，不假定外部现成镜像支持本机架构。
   构建失败记录原因，不能通过删掉难跑测试或换任意 shell 命令冒充原评估协议。
3. 复用同一 task_factory、workspace 与 program；在示例任务层增加 SWE-bench 适配器。
   不沿用人工样例 `src/*.py` 的硬编码限制：按真实仓库清单允许源码修改，保护测试、评估入口和环境配置。
   需要修改受保护构建文件的实例先排除并记录，扩展支持另行说明。
4. 停止 candidate 后以可信 base commit 和冻结文件生成补丁；不相信候选 `.git` 索引、hooks、配置或提交历史。
   保留新增、删除、多文件修改及补丁 hash；超预算、二进制或不支持文件类型明确拒绝。
   大仓库仅导出声明的源码/必要基线检查范围，不能照搬人工样例的整目录尺寸上限。
5. grader 使用匹配基线的全新环境，应用候选 patch，再执行固定版本官方测试脚本与日志解析。
   应用策略依固定评估版本验证；不能直接复制 slime 的多级模糊回退并掩盖基线不一致或部分应用。
   若需多次应用尝试，每次从干净基线开始，并保存具体成功方式及结果。
6. 以官方报告 `resolved` 判定二元成功，保留 F2P/P2P 明细、patch 应用状态及原始测试日志。
   单纯进程 exit 0、模型自述完成、公开测试通过均不能代替该判断。
   测试环境与解析器异常拒绝更新；候选造成的明确语法/运行失败可零分，不能把所有“无测试行”统一归类。

### 5.2 最终验收边界

选型优先对照 slime 已实现的 SWE-bench Verified 路径；具体数据 revision、训练/评估实例列表和 ARM64 镜像清单
在阶段 D 开始前提交给用户确认。本次已确认的两例、固定数据/评估器版本与实际 ARM64 镜像见 D1 记录。
此处不宣称 Verified 自带训练集；使用其实例训练属于选定任务子集实验，必须公开任务划分。

最终用少量真实 SWE-bench 实例验证自主代码修改、冻结补丁、独立官方协议判题及训练消费/发布/再采样链路。
所选实例完整执行其规定测试；不要求遍历整个数据集，也不把模型必须解出某个真实实例作为功能交付门禁。
`resolved=true` 才可声称对应“模型修复通过”；失败补丁如实给零分，参考 patch 仅用于验证评估器正例。
阶段 C 优先完成训练流程功能验收，非零任务优势、梯度和参数变化另列任务学习门禁；若 SWE-bench 小样本全同奖励，
记录零任务优势及实际更新情况，不伪造混合奖励，也不据此声称已学会解决 SWE-bench。
功能冒烟可复用极少量实例，但必须标明训练/评估重叠；需要报告独立评估时才另选不重叠实例。
仅报告实际运行的实例、评分、资源与失败类别，不外推完整 benchmark 得分或泛化收益。

## 6. 文件落点与依赖顺序

路径相对 `hyper_parallel/rl/`；A/B/C 的路径已落地，D 仍为计划。不搬迁现有代码或包边界。

| 路径 | 职责/阶段 |
| --- | --- |
| 新增 `rl/agentic/envs/docker_workspace.py` | 异步、有界 Docker 生命周期、复制与冻结导出；A |
| 新增 `rl/agentic/envs/model_relay.py` | 无网络候选的固定 session 模型通道、有界传输及断连上报；B |
| 新增 `examples/code_agent/task.py`、`prepare_data.py`、`fixtures/`、任务清单 | 人工仓库、基线身份、合法提交及独立评分；A |
| 新增 `docker/Dockerfile.code-agent` | 固定 CLI、CPU 工具、中性 shell 与可写 HOME；A |
| 修改 `rl/agentic/codex/harness.py`、`rl/config.py` | task_factory、执行路径、异步判题和配置校验；B |
| 修改 `rl/agentic/codex/gateway.py`、`rl/agentic/core/program_runner.py` | 管理鉴权/地址、传输失败、有界读写和 session 清理；B |
| 定点修改 `rl/agentic/ds_harness/gateway.py`，必要时其 harness 接线 | 同类 SSE 缺陷和共享 runtime 兼容，不增加仓库 recipe；B |
| 新增 `examples/code_agent/configs/qwen3_4b_code_agent.yaml`、`README.md` | 已验证的最小配置、部署和运行入口；C |
| 新增 `examples/code_agent/swebench_task.py`、对应配置/数据准备及必要 Docker 构建文件 | 官方任务/评估适配与 ARM64 环境；D |
| 新增 `tests/trial/` 中必要合同测试及短实验辅助 | 随各阶段交付；不复制旧测试树 |

现有数据适配、batch、Actor、权重同步原则上复用；发现必须修改时先说明具体合同，不顺带重构。
RL 外旧测试或导航若需要同步，列出确切文件及原因后另行征得用户同意。
A/B 已同步示例 README 与获用户许可的两份导航，B 同步已获许可的 `tests/ut/rl/agentic/agentic_ut.py`；
文档区分接口已接入与真实模型训练尚未验收，不把规划能力写成已支持。

## 7. 阶段、验收门禁与交付

阶段依赖为 A → B → C → D。每阶段均执行“最小正确实现 → 定位问题 → 精简 → 受影响回归 → 文档与交付”，
不能把代码/测试精简积累到最后。下阶段不得绕过本阶段必要门禁；真实模型表现与基础设施状态分别记录。

### A：Docker 工作区与独立评分，无 NPU

- 核对当前源码/安装位置、镜像和 Docker 能力，构建候选镜像；不恢复旧训练镜像或旧 editable 安装。
- 两个缺陷样例 `merge_intervals`、`word_counts`：原始实现各得 0，可信修复各得 1。
  真实容器中覆盖新增/修改/删除、公开测试、停止后导出、独立评分；不把参考修复记为模型成果。
- 验证复制后的属主与写权限、shell 不执行 CANN/npu-smi 初始化、CLI HOME/辅助工具可用。
  保留训练镜像的正常 CANN 初始化。
- 必要负例：保护文件、路径越界/链接/重复条目、尺寸超限、候选超时、Docker 故障、取消及清理。
- 交付：workspace/任务/镜像最小代码、CPU 测试与真实 Docker 证据、清理结果。此时尚不宣称 Agent 训练可用。

### B：接入现有 program 与 Gateway，无 NPU 合同验证

- 接入单一任务入口，正常/失败分支共用 M3 构造器；注册到清理的每一阶段均可归因。
- 验证 candidate 可访问本模型 session，却无法操作管理接口或使用其他 session；管理凭证不泄漏。
- 注入断连、后端超限、注册超时、导出/评分故障与取消；已采样记录保留，整组拒绝更新，所有请求/容器有界收尾。
- CPU 验证 reward 只算一次、所有 call 齐全、真实 tokens/logprobs/mask/版本不变，padding 无副作用。
  共享路径改动后回归 Codex 文本任务、DeepSeek、现有 episode/GRPO/DP 与单轮 code 合同。
- 交付：接口与配置校验、关键负例、受影响回归；mock 通过不代替真实 CLI/容器验证。

### C：可控仓库上的真实模型与 RL 闭环

用户已确认先完成功能流程验收，将模型修复能力和有效任务学习分别报告；后两者不阻塞功能交付。

- 先一次真实 rollout，确认模型实际读源码、写源码、执行公开测试，冻结产物由独立 grader 判题。
  再以 Qwen3-4B、两卡、共卡 native vLLM、两步训练和短评估验证完整链路。
- 实测 CLI system/tools 的输入开销后制定 context、单次输出、调用与墙钟预算；不沿用过紧的 8192/1024 组合。
  `max_episode_tokens` 是每次 prompt+action 上限，`max_turns` 含格式重采样；不等于工具次数。
- 功能门禁允许自然全零或全同奖励，但必须完成两步真实 rollout、优势计算、反向与 optimizer step、权重发布。
  记录实际优势、有限梯度和参数差值；零任务优势、零梯度、零参数变化必须如实报告，不能声称有效学习。
  验证 policy 0→1→2 且第二步采样确用新版本；版本递增本身不证明参数有变化。
  保存逐调用对齐、实际行数/padding/action 数、optimizer steps、评估与 final checkpoint 记录。
- 能力门禁另要求至少一例自主修复得 1；任务学习门禁另要求自然组内混合奖励、非零任务优势、非零梯度及参数变化。
  不得注入奖励、参考补丁或偷偷丢弃失败候选，也不能以 KL/权重衰减引起的变化替代任务学习。
- 用既有 DP 合同回归并结合真实不等调用案例检查零损失补齐；不为产生差异伪造轨迹。
- 交付：可复现 recipe、必要短实验入口、独立的基础设施/模型修复/学习门禁结果，关闭本次服务并清理。
  checkpoint 写出与 reload 分开记录，未测 reload 不作恢复可用声明。
- 用户已允许后续使用本地 Qwen3-30B-A3B、最多四张空闲卡测试模型能力；仍先检查设备健康与占用，
  不把旧 MoE 短上下文结果当作仓库长上下文验收，也不因获得授权而跳过当前功能门禁。

### D：SWE-bench 小规模真实仓库功能验收

- D1 选型及 CPU 基线：确认实例清单、数据/评估器版本、ARM64 环境、源码允许范围与预算；
  对首批实例完成原始代码、参考 patch、F2P/P2P 对照，并核对固定版本官方评估结果。
- D2 模型执行与评分：运行少量真实 Codex 仓库任务，验证自主读改测、补丁与独立 grader；
  如实记录 resolved 或失败，不设解决率门槛，不用人工样例或参考 patch 冒充模型输出。
- D3 训练/评估：以最少必要实例和短步数验证真实任务轨迹进入训练、按真实优势处理、发布和再采样，
  并运行小规模评分；继承 C 的 token、episode、DP、失败传播与清理门禁，记录实际时间/内存/调用预算。
  非零任务学习证据复用 C，SWE-bench 全同奖励按第 5.2 节报告，不强制长时间运行等待成功样本。
- Qwen3-4B 先验证流程；模型未修复成功与协议、评分或训练代码错误分别归因。
  不为追求真实题成功率擅自增加模型规模、用卡或实验时长，也不能降低所选实例的判题标准。
- 交付：SWE-bench adapter、固定实例/环境清单、官方评分报告、训练证据、支持边界与运行说明。
  到此可声明选定真实仓库任务的端到端 code agent 流程功能验收完成；模型解决能力按实际结果单独说明。
  完整 SWE-bench 运行、规模化训练与性能评测不属于本次交付要求。

## 8. 实验与开发约束

沿用 [AGENTS.md](../AGENTS.md)：现有结构不变，必要新增文件按现有职责放置，公共内容变更先征得同意，
新增测试/实验辅助在 `tests/trial/`；原生 Torch，功能确认后每阶段精简，不原样复制旧实现。

- 训练镜像：`hyper-parallel/hyper-rl:v0.22.1rc1-unified-arm64`；权重根 `/home/xhy/Project-hw/models`，
  数据根 `/home/xhy/Project-hw/data`。Qwen3-4B 目录已静态确认存在，运行前仍须验证权重完整与实际导入。
- candidate/grader 镜像单独固定完整身份；不沿用参考 Dockerfile 的旧 digest，也不在每个候选内联网安装 CLI。
  SWE-bench 各仓库依赖层与训练镜像分开管理，构建/下载时按正常网络→8990 代理→显式禁用代理的顺序。
- 正常最多 4 卡，必要时最多 6 卡；8 卡必须事先获得用户同意，按训练/推理/并行实验的同时物理占用合计。
  实验前检查健康和占用；A/B 不申请 NPU。
- 被 kill 可检查并清理后重跑，连续三次中断改为短实验；不得降低原门禁或把中断计为通过。
- 原始产物保存在本仓库忽略的 `output/code-agent/` 下，以阶段/run/episode 区分；记录源码 diff 身份、
  配置、镜像、命令、判题/轨迹/更新证据和资源清理结果。控制敏感数据与凭证进入日志。
- 分别记录“实现”“CPU/Docker 通过”“真实自主修复”“RL 更新”“独立评估”，旧项目证据不能代替新仓库验收。
- 每阶段形成可审查交付，提交按用户授权执行，不自动 push；此前 M1/M2/M3 三个提交保持为已完成迁移历史。

## 9. 进展与下一步

| 项目 | 当前状态 | 下一门禁 |
| --- | --- | --- |
| 范围与方案 | 用户已批准方案与 A/B 实施；真实 SWE-bench 仅要求小规模功能正确 | 后续阶段按已定义范围推进 |
| A Docker 与评分 | 已实现、精简、独立 review；23 项 CPU 及真实 Docker 验收通过 | 进入 B 前保持已验收合同 |
| B program/Gateway | 已实现、精简、独立 review；569 项 CPU 回归及真实 CLI/容器接线通过 | 进入 C，使用真实模型/NPU 验证训练 |
| C 人工仓库 Agent RL | 两卡 4B 两步功能及同拓扑 step 2→3 恢复续训通过，旧测试同步、独立审计与清理完成 | 训练任务学习未观察到，SIGKILL 来源仍未知，D 尚未启动 |
| D SWE-bench | 已确认仅做硬件可承受的小规模功能验收，两例官方正反例通过，官方正反例、语法负例、两卡两步训练与保存通过；30B 两例补充推理均为 0 分 | 官方评分正反例、自主读改测、真实轨迹训练消费与发布再采样 |

### A 验收记录

实现与命令入口：[仓库示例](../examples/code_agent/README.md)。新增 workspace、任务/数据适配、两个公开仓库、
候选 Dockerfile、两份 CPU 合同测试及一个显式 Docker 实验入口；没有修改现有训练/Gateway 运行代码。
主 coordinator 整合、构建与实验；工作区、任务评分和独立 review 分别由子代理并行完成。

| 验证 | 结果 |
| --- | --- |
| CPU 合同 | `test_docker_workspace.py` 13 项 + `test_repository_task.py` 10 项，共 23 项通过 |
| merge_intervals | 缺陷版 2/5，奖励 0；参考修复 5/5，奖励 1 |
| word_counts | 缺陷版 1/5，奖励 0；参考修复 5/5，奖励 1 |
| 真实文件变更 | 新增模块、修改源码、删除 legacy 文件均进入冻结产物与变更清单 |
| 真实负例 | 保护文件修改、符号链接、缺失可执行程序、超时、输出超限、取消，共 6 类通过 |
| 镜像与工具 | 由指定统一镜像离线构建；Codex 0.152.1、真实 patch helper、HOME/属主写权限通过 |
| 资源 | candidate/grader 无挂载、无设备、无网络；1 CPU/1 GiB/64 进程；A 占用 0 NPU |
| 独立复核 | 重新计算四份 tar 的文件及内容 hash，与 manifest/奖励证据一致；无阻塞发现 |

候选镜像：`hyper-parallel/hyper-code-agent:stage-a-unified-arm64`，完整 ID
`sha256:12c0ec8f5b1a8024b415ecab27f26e9bb4d0d5434b175cd9ea8cfb5744d2c6da`。
本地证据根 `hyper_parallel/rl/output/code-agent/stage-a/`：`image-build.log`、`install.log`、
`import-paths.log`、`cpu-tests.log`、`pylint.log`，以及 `docker-run-2/completed.json` 和对应 tar/manifest。
`source-manifest.json` 固定本次实现文件身份，`cleanup.json` 记录两个控制容器已移除，
按两个 run ID 再次扫描无候选/grader/控制容器残留。Pylint、Markdown、目录清单和变更链接检查通过。
导入已核对来自当前 checkout；控制容器为满足既有 RL 导入链只读挂载 NPU 运行库，未映射设备。
最初导入因缺运行库失败，已修复控制环境后重跑；不是实验被 kill，不计为通过或模型错误。

相对旧参考实现的必要适配：每条隐藏用例全新 grader，避免运行期污染；Docker 管理/传输/停止故障与
候选预算失败分开；严格 tar 结束标记与路径校验；取消后收束写入及清理。独立 review 与 lint 后精简完成。
本阶段仅证明环境和独立评分可用，参考修复不是模型成果；模型循环、Gateway、RL 训练和 SWE-bench 均未运行。
后续在本表追加各阶段实际验收与证据位置，不再为每个阶段建立重复总览文档。

### B 验收记录

主 coordinator 负责配置/runtime、无网络模型通道、整合和真实实验；子代理并行负责两套 Gateway、
program 接线及独立 review/通道合同测试。保留既有目录及 A 实现，不引入第四种 runner。

- 单一 `task_factory` 与文本奖励互斥；候选镜像内执行 Codex、冻结后 await 判题、复用 M3 逐调用构造器。
- 独立管理鉴权、跨受信 rank 凭据同步、注册取消清理、幂等删除与迟到注册拒绝；DeepSeek 同步管理合同。
- Gateway 请求/响应/SSE 字节限制、分块读取时限；stdio 通道绑定目标和 session，断连上报不可训练。
- 跨进程候选槽位覆盖独立评分；已归因模型格式失败不运行 grader，基础设施失败保持更高优先级。
- `max_response_bytes` 同时传递到管理客户端，聚合快照超限明确拒绝，不截断 token 或历史。

| 验证 | 结果 |
| --- | --- |
| CPU 全量 RL + trial | 569 项通过，包含既有文本/DeepSeek、episode/GRPO/DP、单轮 code 和 A 合同 |
| 新增 B 合同 | 配置 12、program 12、Gateway 18、relay 6，共 48 项；覆盖鉴权、取消、超限、慢速传输、TCP 断连及跨进程槽位 |
| 真实容器接线 | 固定 Codex 0.152.1 执行读文件/增改删/公开测试，停止后独立 grader；2 次调用归属 1 个 episode，奖励 1 |
| 轨迹结构 | 原样保留测试后端提供的 token ID、action mask、L-1 logprob 和 policy version 7 |
| 原始产物复核 | 独立重算 tar/manifest hash 一致，成功报告在候选/grader 清理后写出 |
| 运行范围 | 0 NPU；真实 Docker/CLI/Gateway/relay/grader，模型响应及 token/logprob 明确为合成测试数据 |

证据根：`hyper_parallel/rl/output/code-agent/stage-b/`。
`cpu-regression-final.log` 保存整合回归，`docker-final/completed.json`、CLI JSONL、Gateway 原始记录及
submission tar/manifest 保存接线证据；`pylint.log` 保存代码检查。
`source-manifest.json` 固定实现身份，`import-paths.log` 核对当前 checkout；`cleanup.json` 记录控制容器已回收，
按本次两个 run ID 再次扫描无残留。Markdown、目录清单、变更链接及 diff 空白检查通过。
首次真实接线失败来自验收脚本未处理参考补丁的删除项，修正后通过；旧 DeepSeek mock 的凭据参数也已定点同步。
这些失败均保留记录，没有按模型零分或成功忽略。

代码检查按镜像 Python 3.12 运行规则；命令仅禁用旧全局 checker 的 `C9002`（禁止直接导入 Torch），
因为它与当前原生 Torch 规则冲突，不修改公共 lint 配置，也未在源码添加规避注释。
无网络转发只支持当前本机 HTTP Gateway 和固定镜像 Python 路径；完整响应有界缓存后交给 CLI，
write/flush 成功不代表消费 ACK。聚合快照预算和已释放 session 标记的生命周期限制需在 C 配置短实验时保留。

**B 证明功能接线正确，不证明模型自主修复或学习。** 本轮没有真实 vLLM 生成、Actor 更新、权重发布或 SWE-bench 实验；
这些仍是 C/D 的验收任务，不以脚本化参考修复替代。

### C 功能验收与实验记录

已新增 `examples/code_agent/configs/qwen3_4b_code_agent.yaml`、`tests/trial/_repository_rollout.py` 和
`tests/trial/_repository_train.py`。初始配置为两卡 DP2/TP1、16K 上下文、2K 单次输出、12 次调用、
两个训练 step 和短评估；实际预算须以成功启动后的真实探针调整，当前不是已验收 recipe。
配置已在当前 checkout 的统一镜像内通过 `resolve_vllm_automatic_limits → build_algorithm → validate_config`。
观察器增加同 prompt 混合奖励、实际动作非零任务优势、非零梯度/参数变化、optimizer steps、最终评估和 checkpoint 校验。
KL 与 weight decay 为零；独立 review、代码风格和 diff 空白检查通过。

2026-09-21 预检：物理 NPU 2、3 健康 OK 且无运行进程，0、1 有 Warning，未选用。
显式映射两张卡的控制容器返回可见设备数 0；只放开 seccomp 的无网络诊断容器仍返回 DCMI 初始化错误 `-8020`。
首次真实 vLLM 启动因此失败，不记为模型失败或实验通过。

自动审批拒绝了旧 M3 使用的 `privileged + host 网络 + Docker socket + 可写仓库挂载` 组合，
理由为两卡实验授权不覆盖该广泛宿主访问权限。已尝试上述更小权限替代方案；用户随后明确允许本次受信训练控制容器采用该配置。
候选/grader 始终保持无特权、无网络、无 NPU；不能绕过审批或用 B 的脚本化结果替代真机验收。
按此次限定授权继续真实探针，再推进两步训练，实际模型设备仍限定物理 NPU 2、3。

授权后 native vLLM 已在物理 NPU 2 启动，使用真实 Qwen3-4B 和候选容器；以下探针均保留原始失败，
尚不能得出自主修复或 RL 学习通过结论。证据根为 `hyper_parallel/rl/output/code-agent/stage-c/`，
每轮 `probe-N/summary.json`、`sessions/*/gateway-events.jsonl`、CLI 事件及存在时的 submission tar/manifest
分别记录候选结果、真实请求与响应、执行命令和冻结产物。

| 观察 | 处理与实现边界 |
| --- | --- |
| probe-1 在调用模型前遇到 Transformers 5 返回 BatchEncoding | 探针显式读取 `input_ids`，异常写入失败状态；不计候选或奖励 |
| probe-2 两个候选分别 3、2 次调用后零分；输出用尽于未闭合 thinking，未修改源码 | `/no_think` 只是提示；修正显式 `reasoning.effort=none` 映射 `enable_thinking=false`，省略/开启行为不变；28 项协议 CPU 测试通过 |
| probe-3 确认实际请求关闭 thinking，但模型反复调用未声明的 apply_patch 并产生非法 JSON | 保留格式失败和全部采样，沿用 M3 归因；统一提示使用已声明的 exec_command 编辑，不注入补丁或修复代码 |
| probe-4 在缺文件反馈后继续猜测路径，耗尽普通调用预算 | 对照原始与转换后请求确认工具观察完整；此类预算耗尽拒绝更新，不改为零分 |
| probe-5 自主写源码并运行公开测试，但写入猜测的新模块，测试仍调用原模块 | 统一补充带路径的公开文件读取，包含入口和公开测试；不能把生成代码或执行测试本身当作修复成功 |
| probe-6–8 读取实际文件后仍循环读文件或执行测试，未完成修复并耗尽调用预算 | 原始工具参数、CLI 执行和返回观察相符；不是 Gateway 丢失反馈，失败仍拒绝更新 |
| probe-9 使用精简系统提示，初始输入降至 2,806 tokens；两题分别 11、12 次调用，奖励均为 0 | merge_intervals 有真实源码 diff，但仅通过 3/5 私有测试；word_counts 为模型格式预算耗尽，不执行 grader；均不证明修复成功或任务学习 |

probe-5 初次启动还遇到示例导入路径问题，修正控制环境后才产生模型调用；日志与修正后的日志分别保留。
探针参数调整统一作用于后续所有候选：明确 thinking 模式、可用工具和实际文件路径，temperature 曾尝试 0.7，后恢复 1.0。
新增可选 `agentic.codex.model_instructions` 使用 Codex 官方
[`model_instructions_file` 配置](https://developers.openai.com/codex/config-reference/)替换内置 base instructions；
不设置时保留 CLI 默认提示。固定 CLI 0.152.1 的 probe-9 原始请求已确认替换生效，仍保留 CLI 附加权限说明。
当前内容仅含通用读改测、工具和保护范围要求，不包含参考解法或私有用例；工具声明和评分合同不变。
旧首次输入实测为 6,404 tokens，其中系统文本 4,178、工具模板增量 1,818；不能将自定义提示的实验结果
写成 Codex 默认提示验收通过。

probe-10 在精简提示上统一尝试 `reasoning_effort=low`、单次输出 4,096 tokens，并移除 `/no_think`。
merge_intervals 完成 4 次调用，修改了未被入口使用的新模块，独立判题 2/5、奖励 0。
word_counts 期间 vLLM 服务退出码 137，Gateway 正确拒绝基础设施中断；没有证据确定为 OOM 或长时间运行导致。
这是首次服务 kill，尚未达到连续三次；`interruption.json` 保存中断信息，不能作为通过或模型零分。
所有探针保持五项私有测试、二元奖励、12 次调用预算及失败分类，不针对单个候选补救、删掉违规产物或选择性保留成功。

最终共享 CPU 回归 576 项通过；受影响文件 Pylint、Markdown 和目录表检查通过，独立 review 未发现接线阻塞。
首次探针收尾时，候选/grader 与受信控制容器均已清理，设备检查无运行进程；复现参数见 `controller-launch.json`。
该轮尚未开始两步训练，没有任务优势更新、发布再采样、短评估或 final checkpoint 通过结论。
用户已确认先用真实自然候选完成功能流程验收，不以 4B 必须修复成功或产生非零任务优势为前提；
能力及任务学习不足单独记录。后续 Qwen3-30B-A3B、最多四张空闲卡的模型能力实验已获授权，无须重复申请。
四卡 MoE 的 M1/M2 证据只证明较短上下文训练，仓库长上下文及最终 checkpoint 仍需重新验收。

功能验收继续使用两卡 4B、精简系统提示、关闭 thinking、2K 输出、每题两候选和 24 次调用预算；
保持 KL/weight decay 为零及所有原始失败分类。训练观察器显式选择 `--acceptance functional`，
默认 `learning` 仍要求修复成功和有效任务学习；新增 8 项门禁测试通过。
`functional-1` 在配置校验时拒绝 mini-batch 4 大于两候选，未开始模型训练；修正为 mini-batch 2 后启动
`functional-2`。运行配置、安装证明、日志及后续验收证据均保留于对应目录。

`functional-2` 的四个真实候选分别产生 13、24、24、9 次调用；其中一个候选反复执行合法 grep，耗尽普通调用预算。
系统按既有合同拒绝整组，未完成第一个训练 step，全部 session 已释放。此运行不是随机 kill，也不能因其他候选
已返回而选择性消费其轨迹；没有 optimizer、发布或 checkpoint 通过结论。

随后按用户已授权范围启动四卡 Qwen3-30B-A3B 实验 `moe-functional-1`，配置入口为
`examples/code_agent/configs/qwen3_30b_a3b_code_agent.yaml`；仍采用功能门禁，不要求正分或非零任务优势。
最终 checkpoint 目标为宿主机 `/dev/shm/hyper-rl-code-agent-c-20260921`：启动前磁盘可用约 246 GiB，
预估完整 checkpoint 约 228 GiB，余量较小；tmpfs 可用约 1,008 GiB、宿主内存可用约 1.1 TiB，故选用 tmpfs。
保留完整保存内容，不以减少参数或 optimizer 状态绕过验收；该位置不保证宿主重启后持久保存。
以上为四卡实验启动时的资源与保存计划，实际运行结果记录如下，不能据此声称 checkpoint 已写出。

用户随后批准仓库普通预算到期后冻结并真实判题。新增结构化终态核对、通道排空、主动停止和最终状态检查；
仅在匹配的预算终态与主动停止证据下，把固定 CLI 的精确 HTTP 409 重试消息保留为诊断。
未知 CLI 错误、超时、输出超限、OOM、传输或停止失败仍优先拒绝；文本/DeepSeek 和模型格式失败合同不变。
预算 episode 的全部调用共享真实奖励，并标记 `truncated=true`、`terminal_reason=max_completions`；
原始 `finish_reason`、prompt/action token、logprob 及 CLI 输出不变，不能伪造最终回答。

预算改动后的全量 CPU 回归 607 项通过。真实 Docker/CLI：`budget-docker-positive-3` 奖励 1、
`budget-docker-negative-2` 奖励 0，另自然结束正例奖励 1；独立 hash 审计一致。
前两次预算正例暴露 CLI 精确重试诊断和 Docker stop 状态竞态，均已修复并保留失败证据。
这些实验使用脚本化模型响应，仅证明预算收尾与独立判题，不计模型能力或 NPU 训练通过。

`moe-functional-1` 外层退出码 137 后由 coordinator 定向停止，未完成更新，未写出验收 checkpoint。
两卡 4B 的 `functional-3` 使用新预算处理，完成第一个训练 step 与 policy 0→1 发布：两 rank 实际调用行
16/35，零损失 padding 19/0，物理行均为 35，各执行 18 次 optimizer step；四个候选自然奖励全零，
任务优势、梯度和参数差值均为零。`acceptance/training.json` 保留证据，不据此声称有效任务学习。
仓库预算冻结判题已在真实模型轨迹触发；第二步实际使用 policy 1 采样，全部 session 已释放。

第二步更新后发布失败：最后 session 于日志时间 11:55:53 释放，两 vLLM worker 于 11:55:57 正常 sleep，
DP1 worker 至 12:00:12 意外退出，随后权重唤醒连接被拒绝。退出原因未知；控制容器 cgroup v1 的
`oom_kill=0`、`failcnt=0` 不足以排除所有外部原因，不能认定为 OOM 或随机 kill。
预算清理只针对自己的 Docker CLI PID 和候选容器，时间上也早于该退出四分钟以上；未发现误杀证据。
因此 `functional-3` 不满足两步完整发布、评估及 checkpoint 门禁，不能标记 C 通过。

`functional-4` 仅将 `max_turns` 从 24 缩短为 8，保留其余配置与门禁，实际完成两个训练 step、policy 0→1→2；
每步两 rank 均为 16 个真实调用行、padding 0，
各执行 8 次 optimizer step。最终评估为 0/2，checkpoint 及 Hugging Face 导出完成。
但本次显式 `--require-uneven-calls` 门禁未满足，验收退出码为 1，因此不记整体验收通过。
随后 `functional-5` 统一将调用预算设为 16，其他配置及门禁不变；历史已有自然 13 次调用的候选，
因此未选择可能过紧的 12 次预算。

`functional-5` 最终退出码 0，`acceptance/completed.json` 的 functional 门禁通过：

| 验证 | 实际结果 |
| --- | --- |
| 真实训练与发布 | 两卡 Qwen3-4B，两个 step，policy 0→1→2，第二步实际采样版本 1 |
| 首步不等调用 | 两 rank 调用数分别为 [10, 5] 与 [16, 12]；真实行 15/28、padding 13/0；padding 零动作 |
| 第二步调用 | 两 rank 均为 [16, 16]，各 32 真实行、padding 0 |
| optimizer | 每 rank 两步分别 14、16 次，共 30 次；调用行切片语义不变 |
| 奖励与任务学习 | 训练八个 episode 全零；七个经过独立 grader，一个模型格式失败；优势、梯度及参数差值为零 |
| 最终评估 | policy 2 下 1/2，训练与评估使用相同两个功能任务，不是独立泛化结果 |
| 保存 | final checkpoint、Hugging Face 导出完成；reload 未运行，不宣称恢复已验证 |

验收配方统一为 16 次模型调用、精简 Codex 系统提示、关闭 thinking、2K 输出和 16K 上下文。
`--acceptance functional` 证明真实训练消费、optimizer、权重发布、DP 补齐与保存流程；
默认 `learning` 门禁仍要求自然混合奖励、非零任务优势、梯度和参数变化，本次未满足。
报告中的 `autonomous_repair=not_observed` 来自训练 episode，不覆盖日志中另行记录的评估 1/2；
评估成功也不能据此归因于训练改善。历史失败全部保留，MoE 仓库实验未通过，D/SWE-bench 尚未开始。
独立审计核验 139 次调用、10 个 session 全部释放及 9 份冻结产物 hash；两个 rank 的 checkpoint
optimizer 状态均为累计 30 步，DCP 存储与 HF 分片完整。对 policy 2 的冻结评估产物再次运行独立 CPU grader：
`merge_intervals` 五项全通过，`word_counts` 五项运行失败，与原评估 1/2 一致；原始 tar/manifest 未改。
成功产物来自模型实际修改源码及运行公开测试，不是注入参考补丁；训练全零仍不证明有效任务学习。
本次两个控制容器和空 MoE scratch 已回收，候选/grader 无残留，NPU 2、3 已释放；实验与 checkpoint 文件保留。
复判明细位于 `functional-5/regrade/`，最终清理记录为证据根的 `cleanup-final.json`。代码尚未提交。

### C 收尾：恢复续训与退出诊断

2026-09-22 按用户授权补做恢复续训及 vLLM 退出诊断。源为 `functional-5/checkpoints/step_2`，
在控制容器只读挂载为 `/resume`；采用相同 Qwen3-4B 和两卡拓扑，目标为真实采样 policy 2、发布 policy 3，
最终评估并保存 step 3。此验收证明状态恢复和继续训练，不证明与未中断运行位级一致或支持跨拓扑恢复。

`tests/trial/_repository_resume.py` 在生产 `checkpoint.begin` 前独立读取期望状态，仅在 trial 中污染
一个 live Actor 本地参数元素；调用原生产恢复后，必须在任何发布前精确核对全部模型、optimizer moments、
scheduler、CPU/device RNG、数据游标及进度。原 checkpoint 不改写，失败立即同步中止，禁止发布污染参数。
为避免 `StatefulDataLoader.state_dict()` 懒建迭代器消耗 RNG，观察器只读核对已恢复的 `next_iter_state`，
再由真实训练迭代推进耗尽游标。

`repair/resume-1/acceptance/restored-rank-{0,1}.json` 已记录两 rank 的恢复前置检查通过：
global step 2、epoch 1、optimizer 参数组 step 30、scheduler `last_epoch=2`/`_step_count=3`，
三个 yielded 游标均为 1。两 rank 首个参数元素被置为 123 后，全模型与 checkpoint 精确相等，
完整 optimizer、RNG 和 loader 也一致；随后正常发布 policy 2，耗尽 loader 推进 epoch 2，已进入 step 3 采样。
`resume-1` 此后因受管 API 父进程遭 SIGKILL 失败，未完成 step 3 发布，不能记为整体验收通过。

生产恢复预检补充 manifest 与进度 step 的整数及一致性检查；独立 checkpoint 单测暴露 RL metrics 导入环，
已将 `episode_rows` 导入延迟到实际使用处解决，不改变统计语义。退出诊断区分受控父进程关闭、子 worker 退出
及信号来源边界；父进程状态不能替代内核 OOM 或其他外部杀进程证据，不自动重试或吞掉发布失败。
`resume-1` 诊断捕获 `return_code=-9`、`parent_signal_number=9`、`shutdown_requested=false`，
证明 API 父进程遭 SIGKILL；施加者及原因未知，不能推断子 worker 收到相同信号或认定 OOM。
容器 cgroup 路径映射已修正，并用实际 CPU 读取验证；`resume-2` 最后存活采样记录 `memory.failcnt=0`、
`oom_kill=0`，正常退出记录为 `return_code=0`、`shutdown_requested=true`。
退出后 `/proc`/cgroup 不可读明确标为 unavailable，不用空值替代活着时采集的证据。

缩短为统一 8 次调用的 `repair/resume-2` 恢复续训通过且退出码 0：同一只读 step 2 checkpoint 再次通过全部恢复前置核对，
真实采样 policy 2、发布 policy 3；每 rank 16 真实调用行、padding 0，执行 8 次 optimizer step，累计 30→38。
训练四个候选及评估两个候选均实际冻结判题；本次训练全零，policy 3 评估 0/2，不证明任务学习。
step 3 checkpoint 和 HF 导出完成，`acceptance/completed.json` 为 passed。
独立审计核对 48 次原始调用、6 个已释放 session 及 6 份产物 hash；物理对齐和零损失 padding 合同继续执行，
本次恢复不额外要求产生不等调用，源码 4B 配方仍为 16 次调用。该结果覆盖同模型、同拓扑和 constant LR
恢复一次，不是未中断运行的位级复现，也不代表首次 SIGKILL 的根因已解决。

全量 CPU 回归 630 项通过；用户随后批准旧生命周期测试的 8 行 monitor mock 与单 owner 调用断言补丁。
定点复跑 vLLM runtime 和退出诊断共 27 项通过，将 `PytestUnhandledThreadExceptionWarning` 提升为错误后仍通过，
原两条 mock 线程 warning 已消除；依赖弃用和 pytest marker 等其他 warning 保留，不宣称所有警告清零。
受信控制容器已回收，两次恢复运行均无候选/grader 残留，NPU 2、3 已释放；清理记录为 `repair/cleanup.json`。
历史 `functional-5` 的 reload 未测仅描述当时状态，当前恢复能力以上述 `resume-2` 真实结果为准。
旧 `consumed_samples/tokens` 字段尚未维护累计统计；本次仅验证其按源值恢复，不将原有零值当作准确计数。
已授权补丁仅修改 `tests/ut/rl/rollout/test_vllm_runtime.py`，原补丁保存在 `repair/pending-runtime-test.patch`；
定点验证见 `repair/mock-regression.log`，无 NPU 临时验证容器已自动回收。C 收尾完成，代码尚未提交。

### D1 已确认选型与环境构建

已按官方数据核实首批候选为 SWE-bench Verified 的 pytest 7.2 两例；此处是实验清单，不是已通过结果。

| 项目 | 固定身份/范围 |
| --- | --- |
| 数据 | `princeton-nlp/SWE-bench_Verified`，revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`，test split |
| 数据文件 | 官方 parquet 2,096,679 bytes，SHA256 `a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd` |
| 官方评估器 | SWE-bench v4.1.0，commit `726c5461e2ef52d83cf1ea2107870a8bb3328d57`；与固定 slime 接口兼容 |
| 首例 | `pytest-dev__pytest-10051`，base `aa55975c7d3f6c9f6d7f68accc41bb7cadf0eb9a`，F2P/P2P 为 1/15 |
| 次例 | `pytest-dev__pytest-10081`，base `da9a2b584eb7a6c7e924b2621ed0ddaeca0a7bea`，F2P/P2P 为 1/63 |
| 共同环境基线 | `572b5657d7ca557593418ce0319fabff88800c73`，官方 Python 3.9 及该版本固定依赖 |
| 镜像计划 | 按固定官方 TestSpec 本地构建 ARM64 依赖/实例环境，再加入固定 Codex 0.152.1；candidate/grader 使用同一完整 image ID |
| 训练控制端 | 沿用已固定的统一 ARM64 训练镜像与 Qwen3-4B，先两卡短流程 |
| 任务划分 | 仅用这两例做功能训练与评估，明确重叠；不作独立 benchmark 或泛化结论 |

本地只有旧 amd64 SWE-bench 实例镜像，不作为当前 ARM64 可用环境。环境构建后记录实际 image ID、
依赖锁定结果和脚本 hash，尚未构建时不虚构 digest。`10081` 涉及 PDB 子进程，必须通过完整官方基线验证，
不能因运行困难裁剪测试或悄悄替换实例。先完成两例原始代码/参考 patch 的官方 F2P/P2P 对照，
再运行真实模型和最短训练闭环；候选始终无网络、无宿主挂载、无 Docker socket、无 NPU。

只读选型证据在 `output/code-agent/stage-d/selection/environment-selection.json`。
原始实例文件含参考 patch、test patch 与私有判题列表，仅供控制端使用，不进入 candidate 镜像或 prompt。
正常主机网络访问超时后，8990 代理成功取得官方数据/源码；未通过关闭 TLS 校验绕过问题。
D1 判题已通过；4B 两轮独立 D2 为空补丁；D3 出现真实语法错误改码，修复后待重试。

用户已明确同意上述两例与环境计划。官方 v4.1.0 cached Conda 锁包含 x86 构建号；
ARM64 构建仅移除第三段平台构建号、将 `ld_impl_linux-64` 换为 `ld_impl_linux-aarch64`，
保留 Python、Conda 包及 pip 包版本，官方 repo/eval 脚本不改动。差异记录于
`output/code-agent/stage-d/build/arm64-port-review/setup-env.diff`。
Codex 运行层提供独立 Python 3.12 bridge；任务仍使用官方 Python 3.9。
原始 build-system 依赖预下载为镜像内 wheel，并记录哈希，使官方 editable 安装可断网运行。

### D1 验收结果

两例均使用固定官方测试脚本与评估器，未裁剪测试。所有指定 P2P 在正反例中均为 `PASSED`，
F2P 在原始代码为 `FAILED`、参考补丁为 `PASSED`；独立 Python 3.11 评估器复判与控制端报告完全一致。

| 实例 | 原始 F2P / P2P | 参考 F2P / P2P | 最终 ARM64 镜像 ID |
| --- | --- | --- | --- |
| `pytest-dev__pytest-10051` | 0/1，15/15 | 1/1，15/15 | `sha256:c89ff36340018972da1a0d2b542fec7d29199d68355139e1ca4030c3771f8fba` |
| `pytest-dev__pytest-10081` | 0/1，63/63 | 1/1，63/63 | `sha256:240f0d5ccbf585dcba4c7e43be00bbcb8a9d5949e33a0526e16ce789fe7fa367` |

控制端 registry 绑定数据身份、完整 image ID、公开基线 tar、官方 eval 脚本与整个 harness Python 源码哈希。
模型 parquet 仅包含公开 issue 与 registry 身份。候选源码由冻结字节对可信基线生成 patch，不信候选 Git；
仅允许 `src/_pytest/`、`src/pytest/` 内 Python 源码，测试、配置及生成的 `_version.py` 受保护。
候选与 grader 使用同一实例镜像；每次 grader 从新容器执行，原始 Git 基线与工作树一致性均检查。
Git、字节码及 egg-info 产物不重放；仍检查 tar 链接、路径和大小，不能借忽略规则逃逸。

本阶段保守边界：官方测试收集失败、必需状态缺失、超时或未知错误拒绝该组，不自动转换成任务零分；
明确非法提交、已证实模型格式失败，以及同版本解释器验证的候选语法错误为零分。
模型完整 prompt/action/logprob 继续逐调用保存，不拼接或裁剪不一致前缀。

主要证据位于 `output/code-agent/stage-d/`：`build/` 的构建与隔离记录、
`registry/completed.json`、`registry/controls-audit.json` 和 `selection/independent-official-regrade.json`。
这些结果证明官方评分正反例与环境有效，不代表模型自主修复或训练通过。

D2 首轮两例各 8 次调用，第二轮为 10051 的 12 次调用；均正常预算冻结，官方判 0，补丁为空。
10051 的首次 sed 未匹配带类型标注的方法，pytest 对源码目录运行未收集测试；
补充公开测试目录及修改后自检指令后，4B 仍重复搜索。此结果只能证明真实调用与判题接线，不能算自主改码。
先继续 4B 两步训练功能验收，再按用户此前允许的四卡 Qwen3-30B 能力测试范围补充真实改码验证。
不同模型的证据分别报告，不将 30B 的能力算作 4B 的训练效果。

交付精简统一保留原有 16 MiB/4096 快照预算。当前全部真实 tar 最多 653 成员、约 4.68 MB，
不需要新的 copy-in 大小配置或任务专属 export 钩子；共享 harness 仅增加受信任务选择完整实例镜像 ID 的能力。

D3 `train-1` 第一轮采样因一个请求的 14,337 输入 + 2,048 输出预留超过 16,384 上限而失败，
未完成 step、策略更新或保存；这是容量错误，不是 kill。所有 session 已释放。
`train-2` 将模型/episode 上限与 4B batched token 容量统一为 32K 后重试，保留全部上下文和动作。
30B 后续 probe 同步使用 32K、每 worker 4 GiB KV，仍限四张物理卡；不通过截断掩盖容量不足。

D3 `train-2` 因一个真实模型 patch 在 `logging.py` 引入 IndentationError 而中止，未完成训练更新。
已补最小负例归因：独立 grader 的 Python 3.9 先确认基线语法合法，严格应用 patch 后 compile 候选变更；
仅明确语法错误返回可训练 0，绑定源码/patch hash、解释器及错误位置，不执行候选源码、不伪造官方 report。
未知收集/导入失败继续拒绝。真实坏 patch 零分、两例 reference 仍官方 1、空 eval 故障仍拒绝，均通过独立复核。
证据为 `syntax-validation/independent-results.json` 和 `selection/syntax-validation-independent-audit.json`；
原始失败运行保持不变。代码冻结后进行 30B 两例推理，再恢复 4B 两步训练验收。

D3 `train-3` 已完成第一组真实官方评分，但训练阶段明确 NPU OOM，未完成 step。
最长实际轨迹 25,256 token；当前 Actor 的全词表 FP32 概率矩阵约 14.29 GiB，与分配错误一致。
微批量已经为 1，不能通过再减 micro batch 解决；故 `train-4` 的功能配方改为正常 6 次调用预算，
保留 32K 推理容量及所有已采样上下文，不截断、不丢弃样本、不跳过零优势更新。
两步训练、策略发布、最终评估和保存要求不变。32K 服务容量不等于两卡能够训练任意 32K 轨迹，
当前未引入长序列词表概率分块算子。OOM 后主动清理失败进程，不能记作外部 kill 或验收通过。

30B 第一例已完成 12 次调用、官方 0 分和空补丁；第二例服务退出码 137，原因未知，未形成奖励，
记为中断。4B 功能验收后只补跑未完成的第二例，不重采样覆盖首例结果。

`train-4` 在完成前主动停止：复核发现验收观察器在 `trainer.train()` 销毁进程组后才读取 rank，
会阻断最终报告。已修为训练前保存 rank，并以模拟 PG 销毁的完整 main 回归验证所有原验收门禁及报告写入。
该轮为协调者主动中止，不计外部 kill 或通过；`train-5` 在清理后使用修复观察器重跑相同六调用配方。

### D3 两卡功能验收结果

`train-5` 使用修复后的观察器退出 0，核心功能门禁全部通过；不能将此结论扩展为模型成功修复或有效学习。

| 项目 | 实际结果 |
| --- | --- |
| 环境 | 固定统一镜像、Qwen3-4B、物理卡 2/3、6 次正常调用预算、32K 推理容量 |
| 训练 | 两步、8 个 episode、48 次真实调用；全部官方判 0，补丁为空 |
| 策略 | `0→1→2`；后续原始 Gateway 记录证实实际使用新版本采样 |
| 优化器 | 每 rank 每步 6 次执行，累计 12；奖励/优势/梯度/观测参数差值全零，未观察到任务学习 |
| 轨迹 | 原始 P/A、action mask 和 rollout logprob 由运行时核验；独立审计复核持久调用、计数及指标 |
| DP | 本轮两 rank 各 12 条真实轨迹，无实际 padding；不将 C 的不等长证据称为 D 的新观察 |
| 最终评估 | policy 2、两例各 6 次真实调用，官方 0/2；train/eval 重叠，不是独立 benchmark |
| 实际长度 | 本轮训练与评估最长 P+A 为 11,233 token，未裁剪原始采样上下文 |
| 保存 | final checkpoint、两 rank runtime 和 HF 导出完成；结构/索引/优化器步数/RNG/游标/调度器独立核验通过 |
| 恢复 | 本阶段未运行 checkpoint reload；不以文件审计替代真实恢复 |

结果入口：`train-5/acceptance/completed.json`、`selection/train-5-independent-training-audit.json`、
`selection/train-5-independent-checkpoint-audit.json`（均位于 `output/code-agent/stage-d/`）。

### D 阶段交付边界

- D1 官方原始/参考正反例、明确语法负例归因、D3 两步训练/发布/评估/保存均已验收。
- 30B 四卡补充推理：10051 取首次运行已完成结果，10081 取中断后单例重试；各 12 次调用，均为官方 0 和空补丁。
  两份独立审计为 `selection/rollout-1-independent-audit.json` 与 `selection/rollout-2-independent-audit.json`。
  不将第一次服务中断记为零分，也不把 30B 推理证据算作 MoE 训练验收。
- 4B 成功训练轮的 10 个 train/eval episode 均为空补丁，没有自然出现语法负例；该分支由先前真实模型坏 patch
  的独立重放及训练观察器合同测试验证，不能声称它在成功训练轮被实际消费。
- 只支持固定两例 pytest Python 源码任务、同实例固定 image ID、16 MiB/4096 快照；不提供任意仓库镜像构建器。
- 训练调用预算为 6，完整保留已采样 P/A；32K 服务容量不是任意 32K 两卡训练保证。长轨迹 FP32 全词表峰值仍是限制。
- 未运行完整 SWE-bench、泛化评测、MoE 仓库训练或本阶段 checkpoint 恢复加载；不宣称修复成功率或任务学习改善。
- 最终 CPU 回归 64 项通过；观察器生命周期修复后专门 6 项通过（与前者有重叠，不相加为独立用例总数）。
  Pylint、目录 catalog、Markdown、新增相对链接及 diff 检查通过。导航既有 10 处失效路径另存检查记录，未扩展修改范围。
- 用户已批准仅同步 RL 外的 `.agent/rules/rl/module-map.md`、`docs/rl-navigation.md`；本阶段未修改公共运行代码。
  本阶段没有提交或推送。

最终控制容器、候选/grader 均已清理，本次使用的物理卡 2/3/4/5 无残留进程；镜像和实验产物保留。
汇总检查见 `output/code-agent/stage-d/delivery-checks.json`，清理与设备状态见 `cleanup.json`、`final-npu-status.log`。
成功短工具 stdout 已抽查在 CLI、Responses、Gateway tool content 和实际采样 token 解码中一致；
CLI 自身对长工具输出可能明确截断，训练保留模型实际接收的上下文，不承诺未交给模型的完整命令输出。

## D 后续：推理空间归因实验

用户已同意先做纯推理对照，再据结果处理训练显存与并行。沿用两例、固定 registry/镜像/官方 grader，
使用本地 Qwen3-30B-A3B 和物理卡 2/3/4/5；本阶段不混入训练更新或私有修复提示。
先验证真实 thinking 两轮工具协议，再比较 thinking 开/关 × 12/24 次调用。
各组统一中性提示（移除 `/no_think` 与具体调用数字）、temperature 1、固定 seed、40,960-token 服务容量及单次 4096 输出预算。
4096 包含思考和动作，若思考被 length 截断，则单列为预算不足，必要时仅作配对扩预算验证。
单实例分别运行，保留中断与错误，检查真实读写测试行为、循环、上下文长度、思考结束、补丁及官方分数。
这是两例探索性定位，不作为成功率或泛化结论；配置和证据在 `output/code-agent/stage-d/diagnosis-1/`。

模型原始 config 声明 `max_position_embeddings=40960` 且无 rope scaling，故本次不强制超范围 64K；
资源增加不会自动改变模型声明的上下文上限。相同 seed 仍受实际工具反馈时间等影响，结果仅作探索性归因。

### 首轮定位：thinking 工具证据误拒

两轮简单工具预检通过，但真实 CLI pilot 的第二次调用触发 `parser_input_mismatch`：
思考中包含两个工具示例，实际 content 只有一个工具调用，旧校验错误地比较整段原文与 Hermes 的 content 输入。
这属于协议误拒，未形成有效候选分数，不能计为模型修复失败。
修复在原 `rl/tool_protocol.py` 中捕获真实 Qwen3 reasoning parser 的输入/结果，绑定同一请求，
再验证 content 与 Hermes 输入一致；完整原始 token/logprob 不变，缺失、歧义或冲突仍拒绝。
无 Hermes 的纯思考输出保持原路径，不宣称所有响应均携带分段证据。

相关 CPU 回归 83 项及 hook/既有插件回归 14 项通过；实际 Qwen3/Hermes parser 重放两条历史响应通过，
不将重放算作原 pilot 成功。修复后的真实 CLI 复测及八组正式对照尚在进行，证据保留于上述诊断目录。

修复后的 CLI 复测确认分段证据真实加载、思考历史保留；第二次动作实际含非法 JSON 转义，
正确归为模型格式错误，额度耗尽后终止，未进入官方判题。
首组 off/12 完成 12 次调用与官方评分，空补丁、0 分；on/12 在第 8 次调用后遭服务退出 137，未评分，
容器 OOM 计数为零，未确认 kill 原因。已有调用无输出截断，最长 prompt 7462 token，不能将此中断归因为上下文不足。
按实际 thinking 调用速度，后续八组在 `extended-wallclock/` 单独重跑，统一将外层 CLI 总时限设为 7200 秒，
其余模型、提示、采样、上下文和输出预算不变；旧运行保留，不选择性拼成成功矩阵。

### 暂停扩展实验，转为完整轨迹审查

用户要求停止扩大实验。调度器、候选和四卡推理服务已停止；新矩阵只完成 10051 的四组，
10081 首组由用户要求中止，不记为零分。停止记录见 `diagnosis-1/operator-stop.json`。
四组均为空补丁、官方 16 项覆盖判 0；off/12、off/24 分别用完调用额度，on/12、on/24 均在第 6 次自然结束。

逐轮审查修正了先前归因：关闭 thinking 时，完整文件读取被 CLI 截为头尾各约 5000 字符，
关键源码未进入下一次实际 prompt，随后重复错误 grep；开启 thinking 的两组从未读源码，
向不存在的 shell session 写拟议 diff、执行错误或未匹配的 sed，未复查文件就声称修改成功。
短工具错误反馈均已核实进入实际 prompt；冻结源码确实未变，不是导出丢失修改。
全库测试缺少可选依赖属于官方选例环境范围，不能与独立官方评分混同。

实际工具说明的默认 10000 tokens 与观测到的 10000 字符截断不一致，当时 CLI 默认来源尚未定位；续接核查见下节。
工具/提示还包含无关 skills、不可用审批参数及与断网容器矛盾的网络说明。
这些事实不足以归因纯模型能力或主要硬件不足；后续先审查输出保留与工具使用合同，
未获新的推进指令前不恢复扩展矩阵或新增训练实验。

### `master-code-agent` 续接：先修长工具输出

固定 Codex CLI 0.152.1 对未收录的 Qwen3 模型使用回退元数据，默认工具历史上限为 10,000 字节；
单次 `exec_command.max_output_tokens` 的说明写 10,000 tokens，但实际取两个上限的较小者。
因此历史 `logging.py` 的 30,010 字符输出在 CLI 事件中完整，在下一次模型请求中仅剩首尾各 5,000 字符。
本分支对仓库任务的隔离 CLI 配置设置 `tool_output_token_limit = 10000`；文本 Codex 路径不变。

本分支新增的无模型真实 CLI/Docker 回归在可控仓库镜像中执行 30,010 字符的文件读取：
CLI 事件保存完整 stdout，下一次 Gateway 原始请求含完整头、中、尾及全文，没有截断标记；
冻结仓库未变，独立 grader 执行 5 项后真实给 0。原可控仓库的合成读改测回归仍得 1。
证据分别在忽略目录 `output/code-agent/cli-limit-20260923-2/` 和
`output/code-agent/cli-limit-regression-20260923-1/`。这验证 CLI 到下一次请求的传输，
仅覆盖一次长输出与两次合成调用；CLI 的未知模型回退上下文为 272K，而现有 vLLM 配方为 16K 或 32K，
多次长读取后的真实上下文容量仍需单独核对。这些无模型实验不证明模型会完成 SWE-bench 修复。

用户批准仓库任务工具合同调整后，隔离 CLI 关闭无关 skills、图像、目标与用户提问工具；
Gateway 只在模型侧移除 `exec_command` 中不可用的审批字段，并将固定 CLI 注入的权限和
环境段改为候选 Docker 的真实约束：`network=none`、无 Docker socket/NPU、`/workspace`
为源码目录、`/tmp` 为临时目录。`original_request` 保留 CLI 原文，实际采样 token 和独立评分合同不变；
若固定 CLI 段落不匹配则按基础设施错误显式失败。`write_stdin` 仍只接受真实存在的进程 session。

候选 `/opt/hyper-codex-home/checked_patch.py` 从 stdin 接受 Codex patch 块，经固定 CLI 的
`apply_patch` 子命令执行；它读取编辑前后文件字节，输出 hash 和有限 diff，零状态却没有文件变化会返回非零。
最终提交仍以冻结快照和独立 grader 为准。固定 CLI/Docker 的合成回归已覆盖长输出、真实编辑反馈及
独立评分：编辑模式的下一次原始请求收到 `PATCH VERIFIED`，冻结补丁非空，故意错误的修复仍由 grader 判 0；
可控参考修复仍判 1。对应忽略目录为 `output/code-agent/tool-contract-long-output-20260923-1/`、
`tool-contract-checked-patch-20260923-1/` 和 `tool-contract-repair-20260923-1/`。
独立 review 后又对 MCP 配置和最终工具名做双重拒绝校验；相关 trial CPU 合同 58 项通过，
固定 CLI/Docker 的编辑回归也在 `output/code-agent/tool-contract-reviewed-20260923-1/` 通过。
仓库任务必须显式设置 `agentic.codex.model_context_window`，且不得大于实际 vLLM `max_model_len`。
CLI 默认在该窗口约 90% 后才自动压缩；普通请求还要为 `rollout.max_new_tokens` 留容量和余量。
16K 服务配 2048 输出的两个示例将 CLI 窗口设为 14336，使压缩早于 vLLM 输入容量上限；
32K/2048 示例保留同值窗口。固定 CLI 对未知模型的 272K 回退窗口不能代表实际模型容量。
固定 CLI/Docker 在 `output/code-agent/tool-contract-context-20260923-1/` 再次通过显式 40960
窗口的合成编辑/冻结/评分链路；这只确认 CLI 接受配置，不是 vLLM 多轮容量验收。
默认自动压缩阈值会随窗口变为 90%；压缩可能额外请求模型并丢失部分工具历史，需在真机轨迹中核查。
真实模型行为与多轮上下文按下述单例继续核对；不得把空补丁改判为环境故障。

### `master-code-agent` 四卡真实单例复测

本分支使用四张健康 NPU（2–5）、Qwen3-30B-A3B、固定 0.152.1 CLI 和
`pytest-dev__pytest-10051` 做一次 12 次调用的真实 rollout。新建控制容器缺固定 SWE-bench evaluator，
第一次预检在模型调用前失败，不记为模型零分；随后只读核对旧受信控制容器的 evaluator 4.1.0
共 46 个源码 hash 与 registry 一致，从该容器连接新服务重跑，证据在忽略目录
`output/code-agent/stage-d/branch-rollout-20260923-1204z/`。

正式单例完成 12 次调用，prompt 长度 1685–15923 token，未触发压缩或上下文错误；冻结 patch
为 0 字节，独立官方 16 项评分为 0。两次 `cat src/_pytest/logging.py` 的 30010 字节输出在下一次
原始和模型侧请求中完整，中段 `get_records`、`clear` 等代码可见。一次错用 `execcommand` 收到真实
`unsupported call` 并在下一轮纠正。其余多次 `rg 'item.stash[caplog_records_key]'` 因方括号作为
正则元字符返回 1，模型重复相同失败搜索，未调用编辑 helper，也未执行测试。因此本次空补丁已不能归因于
长输出截断或工具反馈丢失；更直接的行为是搜索循环耗尽调用预算。三个示例配方已增加 `rg -n -F`
字面搜索、失败后换方法和定位函数后先尝试补丁的通用提示，后续复测需单独报告，不把提示变化当成功证明。

同一四卡服务的第二次单例仅增加上述通用搜索提示（运行身份不同）；12 次调用中 9 次使用
`rg -n -F`，正则误用消失，但仍三次完整读取同一文件，0 次编辑与测试，冻结 patch 为空，
官方 16 项得 0。prompt 峰值 22135，完整源码仍进入下一次请求，无压缩或上下文错误。
配置差异、hash 与逐轮审计在忽略目录 `output/code-agent/stage-d/branch-rollout-20260923-1227z/`。
这说明搜索提示改善了工具语法，但没有使 12 次调用内产生有效编辑，不能单凭两次样本推断训练收益。

将调用上限单独改为 24 的第三次诊断在第 19 次普通模型调用后被 Gateway 拒绝，未冻结、
未官方评分，因此不是模型零分。固定 CLI 0.152.1 此时发出 `request_kind=compaction` 的无工具
摘要请求；先前的普通调用双工具校验误将其判为基础设施故障。Gateway 现按 CLI 原始 metadata
区分普通调用和压缩请求，压缩必须无工具并使用隔离 CLI 固定摘要提示；两类请求均校正候选权限。
摘要调用计入模型预算并保留原始请求与实际 token/logprob；压缩输出最多 1024 token，若后端
以长度截断或空摘要结束，则保存采样证据并明确拒绝该 episode，防止 CLI 用不完整摘要覆盖历史。
固定 CLI 的成功压缩提示仅以精确文案归为诊断，其他未知错误仍为致命。

固定 CLI/Docker 的无模型回归在 `output/code-agent/compaction-contract-20260923-2/`
真实触发 `turn → compaction → turn`，确认 30010 字节工具正文进入摘要请求、摘要进入后续调用，
逐调用 token/action/logprob 映射完整；冻结基线经独立 grader 5 例得 0，无残留容器。
首次回归因成功压缩提示误分类而失败，证据保留在 `compaction-contract-20260923-1/`。
受影响 CPU 合同 84 项通过。中断前 19 次不能用于推断最终补丁；修复后的复测如下。

压缩修复后的第四次同例真机运行完成 24 次普通模型调用，prompt 峰值 18490，未触发压缩；
模型使用分段读取和 `rg -F`，但仍未编辑或测试。冻结 patch 为 0 字节，官方 16 项得 0。
证据在忽略目录 `output/code-agent/stage-d/branch-rollout-20260923-1250z/`。
这验证了本分支正常多轮、冻结和评分路径，但不构成真机压缩路径验收；压缩路径目前由固定 CLI/Docker
合成回归覆盖。增加调用预算没有使这一例从反复检查转向修改。

最后一项工具可用性诊断在提示中提供完整 `checked_patch.py` shell heredoc 模板及六次探索界限。
第一次尝试前四卡 vLLM worker 意外被 SIGKILL（服务 exit 137），Gateway 首轮连接拒绝；该尝试
0 次模型完成、无候选命令或评分，不记为模型零分。宿主内存、容器 OOM 计数及 NPU 占用检查均未发现
持续资源争用，kill 原因仍未知。按本地约定用相同四卡配置重启一次，保持提示、seed、数据和评分不变，
仅换运行身份重新执行。重试完成 12 次调用；第六次已定位 `self.handler.reset()`，但后续仍反复
`rg/head/tail`，0 次编辑与测试，冻结空 patch，官方 16 项得 0。提示确实进入模型请求，
其未执行不能归因于提示缺失。失败与重试证据分别在忽略目录
`output/code-agent/stage-d/branch-rollout-20260923-1259z/`、
`output/code-agent/stage-d/branch-rollout-20260923-1303z/`。
本轮新建推理服务和控制容器已定向移除，2–5 卡无残留进程，旧诊断控制容器保留；
本轮 session 的临时 `auth.json` 已删除，逐轮请求、命令、冻结产物和评分证据保留。

综上，历史首要基础设施缺陷（CLI 长输出截断）与本分支发现的压缩请求/提示误归因均已修复并有
真实 CLI/Docker 合成验证；这批 30B 真实样本的空补丁现主要表现为模型持续选择读取/搜索，
即使相关源码、失败反馈和可执行编辑命令均已可见。没有观察到有效 SWE-bench 修复或学习收益。

### DeepSeek Harness 并行接线与上下文合同

仓库任务现可选择 `agentic.runner=deepseek`、`agentic.deepseek.task_factory`，与文本任务的
`reward_callable` 二选一。沿用相同 `RepositoryTask` / `SWEBenchTask`、固定镜像 ID、
Docker 候选、控制端 relay、停止后归档和独立 grader。受信控制端只运行 SDK 管理进程；
实际 DS runtime 由 SDK 的固定 `launch_args_override` 在无网络候选容器执行，
`DSH_CWD=/workspace`，session/home 位于不导出的路径。Gateway 为 DS 固定转发
`/v1/chat/completions`，逐调用原始请求、实际采样 token 和 logprob 沿用现有 episode GRPO。
仓库普通调用预算达到后，结构化 HTTP 409 只有与 SDK 终态及 Gateway 调用数完全吻合才允许
冻结评分；其他 SDK、传输或推理服务故障拒绝训练。旧文本 DS 预算合同不变。

候选镜像使用离线 `Dockerfile.swebench-ds` 复制固定 0.1.1rc1 runtime；后续独立 Cordis
派生镜像只暴露 `bash`，把单流 stdout/stderr 上限设为 4 KiB，并挂载 DS 自有 token meter
与基础压缩插件，固定实际 16K 上下文、提前触发摘要。模型可在 `bash.command` 中使用
`checked_patch.py` heredoc；真实 CPU 回归确认 patch hash/diff、公开测试结果入模与冻结评分。
默认 stock DS 的 64 KiB 尾截断会让过宽工具输出在下一调用超过 16K；4 KiB 上限能
减缓增长，但在长会话中仍可能超限。压缩插件的 token meter 在高熵内容上可能低估，
不能把配置阈值视作普遍容量保证；推理 HTTP 400 仍是不可训练基础设施失败。
当前真实 Qwen3-30B-A3B 轨迹、官方评分和剩余模型动作选择问题见
[交接文档](code_agent_handoff.md#deepseek-harness-对照工具输出与下一步动作)。

### 外部 API 推理合同与当前链路验收

2026-09-28，经用户确认，新增外部 API 的显式仅推理入口。其动机是用已证明能修复该实例的
Qwen3.8 服务验证当前 Code Agent 的工具、产物与评分链路，而不是为训练补造缺失的概率证据。
Gateway 保留现有 Responses→Chat 和仓库模型上下文合同，只在受信端选择推理模式时不请求训练
token/logprob 字段；结构化工具参数、错误传播及请求记录仍校验。后端鉴权只在控制端处理，
错误详情有界脱敏，不能把密钥或未经处理的认证响应写进候选和日志。

`CodexAgentProgram.run_inference()` 与训练 `run()` 共用已有仓库执行、停止、归档和评分生命周期，
但返回独立 `RepositoryInferenceResult`。推理版本为 `None`，训练入口和轨迹构造器显式拒绝推理记录。
有限调用预算的终止解析也按模式区分，不能为推理放宽训练策略版本断言。
产品入口、配置和复现方法统一维护在[仓库示例](../examples/code_agent/README.md#外部-api-仅推理验收)。

当前 CLI 0.152.1 的双工具/checked_patch 路径已完成两个可控仓库和两个 SWE 实例的真实修复，
CPU 合同、真实 Docker 故障回归、固定 CLI 无训练 token 合成验证亦有独立记录。
具体运行配置、逐题分数及证据目录见[当前验收](code_agent_handoff.md#外部-api-仅推理验收2026-09-28)。
这些结果验证推理功能，不扩展训练模型清单，也不替代 Actor/rollout 权重同步、梯度或学习验收。

### 后续训练接入决策（2026-09-29）

用户决定下一阶段优先 Qwen3.8，首版限定 FSDP、训练 TP=CP=1、full_gather 权重同步，
同步期间暂停 rollout。公共 AutoModel、qwen3_5 模型注册与 GDN 分片声明已存在；
主要缺口是 RL family/构建路由、模型专用假设、HF 与推理引擎权重映射，以及真机训练正确性验证。
此处只记录开发方向，未实现新模型训练，也未以 API 推理结果替代训练验收。
完整接手文件、资源和分阶段验收标准统一见[交接下一阶段](code_agent_handoff.md#7-下一阶段qwen38-rl-的最小接入范围)。
