# Hyper-RL Module Map

Use this index to locate ownership; all code paths below are relative to the repository root.
Interface semantics live in product docs. Feature-to-config/code/metric/test traces live in
[feature navigation](../../../docs/rl-navigation.md); project composition lives in
[architecture](../../../docs/rl-architecture.md). Start work with [the RL rule](../hyper-rl.md).

| Area | Code or entry | Detail |
| --- | --- | --- |
| Design and boundaries | `hyper_parallel/rl/rl/config.py`, `hyper_parallel/rl/rl/roles/model_setup.py` | [Design](../../../hyper_parallel/rl/docs/design.md), [feature navigation](../../../docs/rl-navigation.md) |
| Roadmap | — | [Current acceptance and next steps](../../../hyper_parallel/rl/docs/code_agent_handoff.md) |
| Config and CLI | `hyper_parallel/rl/rl/config.py`, `hyper_parallel/rl/train_rl.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md), [feature navigation](../../../docs/rl-navigation.md) |
| Synchronous orchestration | `hyper_parallel/rl/rl/trainer.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Persistence and evaluation | `hyper_parallel/rl/rl/checkpoint.py`, `hyper_parallel/rl/rl/evaluation.py` | [Checkpoint and evaluation contracts](../../../hyper_parallel/rl/docs/architecture.md) |
| Prompt and experience data | `hyper_parallel/rl/rl/dataset/data_source.py`, `hyper_parallel/rl/rl/dataset/contracts.py`, `hyper_parallel/rl/rl/dataset/batch_builder.py`, `hyper_parallel/rl/rl/dataset/episodes.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Algorithms and rewards | `hyper_parallel/rl/rl/algorithm/`, `hyper_parallel/rl/rl/registry.py` | [Feature navigation](../../../docs/rl-navigation.md), [algorithm contracts](../../../hyper_parallel/rl/docs/architecture.md) |
| Actor, Reference, Critic | `hyper_parallel/rl/rl/roles/policy/actor.py`, `hyper_parallel/rl/rl/roles/policy/critic.py` | [Role contracts](../../../hyper_parallel/rl/docs/architecture.md) |
| Training model and optimizer factories | `hyper_parallel/rl/rl/roles/model_setup.py` | [Model construction boundaries](../../../docs/rl-architecture.md) |
| Qwen3.8 integration | `hyper_parallel/rl/docker/`, model setup, native vLLM and weight sync boundaries | [Development scope and acceptance](../../../hyper_parallel/rl/docs/qwen3_8_development.md); base training, native inference and GSM8K RL integration completed, including checkpoint save/restore; full Code Agent training remains unaccepted after Actor backward OOM |
| Process and service cleanup | `hyper_parallel/rl/rl/process_cleanup.py` | [Runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Qwen3 adapters and RL construction compatibility | `hyper_parallel/models/qwen3/adapter/`, `hyper_parallel/rl/rl/roles/qwen3_builder.py` | [Model construction boundaries](../../../docs/rl-architecture.md); main-project callers use the shared AutoModel builder |
| Training-inference consistency model adapters | `hyper_parallel/rl/rl/roles/rollout/consistency_models/qwen3/`, `hyper_parallel/rl/rl/roles/rollout/vllm_plugin.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Generation and topology | `hyper_parallel/rl/rl/roles/rollout/vllm.py`, `hyper_parallel/rl/rl/roles/rollout/topology.py`, `hyper_parallel/rl/rl/roles/rollout/worker.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Agentic contracts and environment | `hyper_parallel/rl/rl/agentic/core/`, `hyper_parallel/rl/rl/agentic/envs/environment.py`, `hyper_parallel/rl/rl/agentic/tools/` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md) |
| External Agent programs and MCP | `hyper_parallel/rl/rl/agentic/codex/`, `hyper_parallel/rl/rl/agentic/ds_harness/`, `hyper_parallel/rl/rl/agentic/core/program_runner.py`, `hyper_parallel/rl/rl/agentic/mcp_server.py` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md); DeepSeek here names the harness, not a supported model family |
| Tool evidence and trainability | `hyper_parallel/rl/rl/tool_protocol.py`, `hyper_parallel/rl/rl/roles/rollout/vllm_plugin.py` | [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md#工具失败归因与收尾) |
| External-API repository inference | `hyper_parallel/rl/examples/code_agent/inference.py`, `hyper_parallel/rl/rl/agentic/codex/` | [Inference contract](../../../hyper_parallel/rl/examples/code_agent/README.md#外部-api-仅推理验收); shared repository lifecycle and grading, no training trajectories or published policy version |
| GSM8K examples | `hyper_parallel/rl/examples/gsm8k/`, `hyper_parallel/rl/examples/gsm8k/configs/` | [RL README](../../../hyper_parallel/rl/README.md), [Agentic RL](../../../hyper_parallel/rl/docs/agentic_rl.md) |
| Single-turn code tasks | `hyper_parallel/rl/examples/code/`, `hyper_parallel/rl/examples/code/configs/` | [Code example](../../../hyper_parallel/rl/examples/code/README.md); private stdio tests, remote SandboxFusion judging, and reviewed data preparation |
| Repository workspaces and grading | `hyper_parallel/rl/rl/agentic/envs/docker_workspace.py`, `hyper_parallel/rl/rl/agentic/envs/model_relay.py`, `hyper_parallel/rl/examples/code_agent/` | [Repository example](../../../hyper_parallel/rl/examples/code_agent/README.md), [development contracts](../../../hyper_parallel/rl/docs/code_agent_development.md), [current acceptance and budgets](../../../hyper_parallel/rl/docs/code_agent_handoff.md); isolated candidate/grader workspaces, independent grading, training and resume entries |
| Publication lifecycle | `hyper_parallel/rl/rl/roles/weight_sync/config.py`, `hyper_parallel/rl/rl/roles/weight_sync/sync.py`, `hyper_parallel/rl/rl/roles/weight_sync/transfer.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Layout, packing and transport | `hyper_parallel/rl/rl/roles/weight_sync/layout.py`, `hyper_parallel/rl/rl/roles/weight_sync/model_adapter.py`, `hyper_parallel/rl/rl/roles/weight_sync/packed_weight.py`, `hyper_parallel/rl/rl/roles/weight_sync/ipc.py`, `hyper_parallel/rl/rl/roles/weight_sync/hccl.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| vLLM update endpoints | `hyper_parallel/rl/rl/roles/weight_sync/vllm_client.py`, `hyper_parallel/rl/rl/roles/weight_sync/vllm_worker.py` | [vLLM rollout](../../../hyper_parallel/rl/docs/vllm_rollout.md) |
| Consistency | `hyper_parallel/rl/rl/consistency/` | [Qwen3 consistency](../../../hyper_parallel/rl/docs/qwen3_training_inference_consistency.md) |
| Metrics and logging | `hyper_parallel/rl/rl/utils/monitoring/` | [Feature navigation](../../../docs/rl-navigation.md), [runtime architecture](../../../hyper_parallel/rl/docs/architecture.md) |
| Runtime installation | `hyper_parallel/rl/docker/README.md`, `hyper_parallel/rl/docker/install_runtime.sh`, `hyper_parallel/rl/docker/patches/`, `hyper_parallel/rl/pyproject.toml` | [Runtime image](../../../hyper_parallel/rl/docker/README.md) |
| UT | `tests/ut/rl/` | [Feature contracts and validation](../../../hyper_parallel/rl/docs/moe_code_agent.md#单元测试) |
| ST and shared recipes | `hyper_parallel/rl/tests/st/test_rl_st.py`, `hyper_parallel/rl/tests/st/test_feature_st.py`, `tests/common/rl_st_cases.py` | [Standalone ST](../../../hyper_parallel/rl/README.md#系统测试); temporarily outside the main-project PR gate |

The runtime supports Qwen3 dense GRPO/PPO, Qwen3-MoE GRPO and Qwen3.5-family text GRPO under the
[documented rollout limits](../../../hyper_parallel/rl/docs/vllm_rollout.md).
MoE construction uses the shared `hyper_parallel/models/qwen3_moe/` recipe; RL weight conversion remains under
`hyper_parallel/rl/rl/roles/weight_sync/`. Shared-project changes follow their own module rules.
Packaging changes belong to the root `setup.py` and `MANIFEST.in`.

Qwen3.5-9B and Qwen3.8 use the existing `hyper_parallel/models/qwen3_5/` text training implementation.
RL identity and namespace mapping live in `roles/model_setup.py`; native text publication completeness lives in
`roles/weight_sync/vllm_worker.py`. The first recipe is
`hyper_parallel/rl/examples/gsm8k/configs/qwen3_8_27b_gsm8k_vllm.yaml`, with CPU contracts in
`tests/ut/rl/trainer/test_qwen3_5_runtime.py` and real acceptance in
`hyper_parallel/rl/tests/st/_qwen3_5_train.py` (`RL_ST_QWEN3_5_CONFIG`).
The 9B repository recipes are `examples/code_agent/configs/qwen3_5_9b_{code_agent,swebench}.yaml`;
their CPU contracts live in `tests/ut/rl/trainer/test_qwen3_5_code_agent.py`.
Complete next-token probability projection belongs to
`hyper_parallel/models/qwen3_5/adapter/selected_log_probs.py`. RL binds this model capability in
`roles/model_setup.py` and dispatches through the normal root call in `roles/policy/actor.py`.
Model numerical contracts live in `tests/ut/auto_models/models/qwen3_5/test_selected_log_probs.py`;
RL boundary contracts live in `tests/ut/rl/trainer/test_token_log_probs.py`.
