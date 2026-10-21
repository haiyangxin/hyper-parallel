# Qwen3-4B / GSM8K 性能实验工作区

本目录保留 Hyper-RL 与 verl 对照中的可复用工具、优化台账和精简结果。
模型、数据集、逐回答日志、checkpoint、profiler、宿主清理策略和 SSH 凭据不进入 Git。
历史原始实验目录仍在开发机器的 `hyper_parallel/rl/output/`，按需通过 rsync 归档。

当前状态：推理、训练驻留和 Actor 动态分批已完成历史验收；old/reference 动态前向通过 CPU 检查，
本地 NPU 对照因外部占卡中断，随后清理输出遇到文件权限错误，未取得有效对比。
新整理的便携启动器尚未进行远端 NPU 验收；提交不代表这项性能优化已经有效。
参见[优化跟踪](optimization_tracking.md)。

## 本地开发与远端运行

1. 本地修改、测试、提交并推送实验分支到 fork。远端通过 Git 获取同一提交。
2. 在远端干净 checkout 启动实验。启动器用 `git archive` 保存独立源码快照，两组使用相同提交及镜像 ID。
3. 训练在 tmux 中运行，日志写入指定结果目录。实验运行期间不要修改该目录下的 `source/`。
4. 本地通过 rsync 拉取结果；继续开发后使用新的提交和新的实验目录，避免覆盖正在运行的快照。

以下命令中的路径均需设置为目标机器的实际路径；源码同步走 Git，日志同步单向拉回，不使用 `--delete`。

```bash
# 本地：配置 SSH 目标后读取远端已有实验日志。
export RL_REMOTE=xhy@127.0.0.1
export RL_SSH_PORT=2233
export RL_REMOTE_RESULTS=/absolute/remote/path/to/results
export RL_LOCAL_RESULTS=/absolute/local/path/to/results
mkdir -p "$RL_LOCAL_RESULTS"
rsync -a --partial --exclude 'source/' \
  -e "ssh -p $RL_SSH_PORT -i $HOME/.ssh/id_ed25519" \
  "$RL_REMOTE:$RL_REMOTE_RESULTS/" "$RL_LOCAL_RESULTS/"
```

私钥保留在本地，不传到远端或仓库。活跃日志的拉取只是当前快照，完成后再同步一次以获取完整结果。
已有 `result.json` 才表示对应 case 通过分析检查，不能仅凭容器退出或日志出现 loss 判定完成。

## old/reference 动态分批：两组各一步

仅比较固定每卡两条与 old/ref 各 4096-token 动态预算。
Actor 动态预算保持 4096，Actor/优化器常驻 NPU、Reference CPU offload；每步 128 题 × 8 回答。
两组都从原始权重开始，使用相同训练 seed 和引擎 seed，关闭 rollout log-prob 返回、验证和 checkpoint。
不追加预算扫描、长训练或其他 NPU 消融。独立采样不保证逐 token 相同，性能必须结合有效 token 工作量解释。

远端前提：8 张 910B、Docker、宿主 Python 3.9+、`npu-smi` 和 tmux，镜像及数据已在本地。
数据目录需包含与原实验相同格式的 `train.parquet`、`test.parquet`；模型目录包含完整 HF 权重和 tokenizer。
同一实验目录恢复时要求相同提交、镜像 ID、模型路径和训练 parquet 哈希；模型目录必须保持只读、不被其他进程替换。
若要复现历史数据顺序，迁移历史精度实验生成的两个 parquet，而不是重新抽样。

```bash
# 在远端仓库根目录、目标提交的干净 checkout 中执行。
export RL_MODEL=/absolute/path/to/Qwen3-4B
export RL_DATA=/absolute/path/to/prepared-gsm8k
export RL_RESULTS=/absolute/path/to/new-experiment
mkdir -p "$RL_RESULTS"
tmux new-session -d -s hp-frozen-forward \
  "python3 hyper_parallel/rl/examples/gsm8k/experiments/frozen_forward/run.py \
  --model '$RL_MODEL' --data '$RL_DATA' --results '$RL_RESULTS' \
  > '$RL_RESULTS/queue.log' 2>&1"
```

启动器默认使用 [RL 运行镜像](../../../docker/README.md)，也可通过 `--image` 指定已校验的镜像。
每组新建独立容器，在源码快照上安装 editable 主包及 RL 插件，核对导入路径。
启动时将已验证的 `q.record_stream(current_stream)` 补丁应用到该实验容器的依赖中；
仅接受已记录的原版/修复版源码 SHA，未知版本明确失败。不会修改宿主安装或 verl 仓库，也不将补丁验证视为原版 verl baseline。
补丁位于 [docker/patches](../../../docker/patches/ascend-sampler-record-stream.patch)，容器内的应用记录保存在 `sampler.json`。

启动器等待连续三次设备空闲和每卡至少 54 GiB 可用，要求 `device_share=false`。
每组运行时记录各卡 HBM、AICore 和外部进程，外部争用或监控失败只停止本实验容器并退出队列。
它不修改远端宿主的定期清理规则；如远端也有清理任务，须先确认其策略并配置仅覆盖实验容器的临时豁免。
旧机器的 root 清理脚本、白名单和硬编码容器 ID 没有迁移到这里。

暂停与恢复：

```bash
# 远端：创建暂停标记；监控器停止当前组，之后的组不会启动。
touch "$RL_RESULTS/PAUSE"

# 明确决定恢复后，删除暂停标记并用原参数再次启动。
rm "$RL_RESULTS/PAUSE"
```

每组成功后保留 `result.json`；再次运行跳过成功组。已运行成功但分析失败时先重试分析，不重跑训练。
中断输出移动到 `interrupted/`，不递归删除容器 root 写入的文件，解决此前重试的权限问题。
这些目录不进入最终统计。容器保留为停止状态，便于读取日志；检查 `container_id.txt` 后可自行删除本实验容器。

## 结果与解释边界

`progress.json` 显示队列状态；每组保存 `config.yaml`、`train.log`、`sampler.json`、`exit_code`、
`phases-rank-*.jsonl` 和 `result.json`。总表为 `performance_report.txt`、`comparison.json`。
`run_manifest.json` 和 `source.tar` 标识源码与运行时；`device-monitor.jsonl` 保存宿主监控。

计时对 old/ref、Actor 更新和发布阶段加同样的设备同步，报告最慢 rank；
有效概率的有限性检查及输入摘要计算放在阶段计时外。服务启动单列，首步总耗时不作为稳态速度。
训练进程 allocated/reserved 峰值与整卡 HBM 采样峰值分别报告，AICore 利用率不是 MFU。
短阶段没有有效采样时使用 null；采样不能排除瞬时峰值或间隔内短暂争用。
只验证一次完整更新，不宣称长期收敛、修复后50步精度或多次重复统计显著性。

日志同步后，可在本地重新生成报告，不加载 Torch 或 NPU：

```bash
python3 hyper_parallel/rl/examples/gsm8k/experiments/frozen_forward/analyze.py \
  --results "$RL_LOCAL_RESULTS"
```

若需要重新审核已完成的一组，加 `--case baseline` 或 `--case dynamic`。
历史汇总和图像见 [results](results/)，所有历史50步结果均早于采样器生命周期修复。
