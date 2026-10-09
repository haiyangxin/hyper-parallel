# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Real NPU worker for the native Qwen3.5/Qwen3.8 hybrid training path."""

import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.checkpoint.state_dict import StateDictOptions
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig

from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import get_model_state_dict


def _build_model(
    dtype: torch.dtype,
    activation_checkpoint: Optional[str] = None,
    distributed_setup: Optional[DistributedSetup] = None,
    checkpoint_path: Optional[str] = None,
    linear_num_key_heads: int = 1,
    attention_implementation: str = "eager",
    full_attention_config: Optional[dict] = None,
):
    """Keep real GDN head dimensions and its three-to-one value/key ratio."""
    full_attention_config = full_attention_config or {}
    config = Qwen3_5TextConfig.from_dict({
        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": full_attention_config.get("num_attention_heads", 2),
        "num_key_value_heads": full_attention_config.get("num_key_value_heads", 1),
        "head_dim": full_attention_config.get("head_dim", 16),
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_num_key_heads": linear_num_key_heads, "linear_num_value_heads": 3 * linear_num_key_heads,
        "layer_types": ["linear_attention", "full_attention"], "pad_token_id": 0,
        "max_position_embeddings": 128, "use_cache": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
    })
    kwargs = {
        "torch_dtype": dtype, "attn_implementation": attention_implementation,
        "activation_checkpoint": activation_checkpoint, "distributed_setup": distributed_setup,
    }
    if checkpoint_path is not None:
        return HyperAutoModelForCausalLM.from_pretrained(checkpoint_path, local_files_only=True, **kwargs).train()
    return HyperAutoModelForCausalLM.from_config(config, **kwargs).train()


def _forward(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Use explicit positions and disable mutable inference state."""
    mask = input_ids.ne(0)
    return model(
        input_ids=input_ids, attention_mask=mask.long(),
        position_ids=(mask.long().cumsum(-1) - 1).clamp_min(0), use_cache=False,
    ).logits


def _loss(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Compute actual masked next-token likelihood in FP32."""
    selected = _forward(model, input_ids)[:, :-1].float().log_softmax(-1)
    selected = selected.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    valid = input_ids[:, :-1].ne(0) & input_ids[:, 1:].ne(0)
    return -selected[valid].mean()


def _assert_finite_gradients(model) -> None:
    """Require gradients for both attention types and their real projections."""
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient for parameter={name}"
        assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient for parameter={name}"
    for name in ("A_log", "dt_bias", "conv1d.weight", "in_proj_qkv.weight", "out_proj.weight"):
        gradient = model.model.layers[0].linear_attn.get_parameter(name).grad
        magnitude = gradient.float().abs().sum().item()
        assert magnitude > 0, f"Expected nonzero gradient for parameter={name}, magnitude={magnitude}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("attention_implementation", ["eager", "sdpa"])
def test_qwen35_training(dtype: torch.dtype, attention_implementation: str) -> None:
    """Validate gradients, independent rows, real recomputation and resumed update."""
    torch_npu.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(7)
    torch.npu.reset_peak_memory_stats()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 0, 0]], device="npu")
    native_options = {
        "attention_implementation": attention_implementation,
        "linear_num_key_heads": 16,
        "full_attention_config": {"head_dim": 256, "num_attention_heads": 24, "num_key_value_heads": 4},
    }
    model = _build_model(dtype, **native_options)
    assert model.config._attn_implementation == attention_implementation, (
        f"Expected requested HF attention={attention_implementation}, actual={model.config._attn_implementation}"
    )
    assert next(model.parameters()).device.type == "npu", f"Expected NPU, got={next(model.parameters()).device}"
    tolerance = {"rtol": 2e-2, "atol": 5e-4} if dtype == torch.bfloat16 else {"rtol": 2e-4, "atol": 2e-5}
    batch = _forward(model, ids)
    short = ids[1:2, :4]
    independent = _forward(model, short)
    torch.testing.assert_close(batch[1, :4], independent[0], **tolerance)
    left_padded = torch.tensor([[0, 0, 8, 9, 10, 11]], device="npu")
    torch.testing.assert_close(_forward(model, left_padded)[0, 2:], independent[0], **tolerance)
    _forward(model, ids[:1])
    torch.testing.assert_close(_forward(model, short), independent, **tolerance)

    checkpointed = _build_model(dtype, activation_checkpoint="full", **native_options)
    checkpointed.load_state_dict(model.state_dict())
    calls = []
    handle = checkpointed.model.layers[0].register_forward_pre_hook(lambda module, inputs: calls.append(None))
    original_loss = _loss(model, ids)
    recomputed_loss = _loss(checkpointed, ids)
    torch.testing.assert_close(recomputed_loss, original_loss, **tolerance)
    original_loss.backward()
    recomputed_loss.backward()
    handle.remove()
    assert len(calls) >= 2, f"Expected GDN recomputation, observed forward calls={len(calls)}"
    _assert_finite_gradients(model)
    _assert_finite_gradients(checkpointed)
    for actual, expected in zip(checkpointed.parameters(), model.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, **tolerance)
    del checkpointed

    before = model.model.layers[0].linear_attn.in_proj_qkv.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=False, fused=False)
    optimizer.step()
    updated = model.model.layers[0].linear_attn.in_proj_qkv.weight
    assert not torch.equal(updated, before), f"Expected GDN update, dtype={dtype}"
    optimizer.zero_grad(set_to_none=True)
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        state_path = Path(directory) / "optimizer.pt"
        torch.save(optimizer.state_dict(), state_path)
        restored = HyperAutoModelForCausalLM.from_pretrained(
            directory, torch_dtype=dtype, attn_implementation=attention_implementation, local_files_only=True
        ).train()
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-2, foreach=False, fused=False)
        restored_optimizer.load_state_dict(torch.load(state_path, map_location="npu", weights_only=True))
    torch.testing.assert_close(_forward(model, ids), _forward(restored, ids), **tolerance)
    for current_model, current_optimizer in ((model, optimizer), (restored, restored_optimizer)):
        _loss(current_model, ids).backward()
        _assert_finite_gradients(current_model)
        current_optimizer.step()
    assert tuple(model.state_dict()) == tuple(restored.state_dict()), (
        f"Restored keys={tuple(restored.state_dict())}, expected keys={tuple(model.state_dict())}"
    )
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, restored.get_parameter(name), **tolerance)
    torch.npu.synchronize()
    result = {
        "runtime": {name: version(name) for name in ("torch", "torch-npu", "transformers")},
        "dtype": str(dtype), "scope": "native tiny hybrid text model; not FSDP or real 27B acceptance",
        "attention_implementation": attention_implementation,
        "resolved_attention_implementation": model.config._attn_implementation,
        "model_type": model.config.model_type, "text_namespace": "model.* and lm_head.*",
        "full_attention_head_dim": model.config.head_dim,
        "full_attention_query_heads": model.config.num_attention_heads,
        "full_attention_key_value_heads": model.config.num_key_value_heads,
        "linear_attention_key_heads": model.config.linear_num_key_heads,
        "linear_attention_value_heads": model.config.linear_num_value_heads,
        "full_attention_query_kv_ratio": model.config.num_attention_heads // model.config.num_key_value_heads,
        "loss": original_loss.detach().float().item(), "head_dim": 128, "value_key_head_ratio": 3,
        "finite_gradient_parameters": len(tuple(model.parameters())), "recompute_forward_calls": len(calls),
        "padding_and_row_isolation": True, "checkpoint_next_update": True,
        "peak_allocated_bytes": torch.npu.max_memory_allocated(), "tolerance": tolerance,
    }
    print(json.dumps(result))
    output = os.environ.get("HP_QWEN35_ST_OUTPUT")
    if output:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        filename = f"tiny-{attention_implementation}-{str(dtype).removeprefix('torch.')}.json"
        (destination / filename).write_text(json.dumps(result, indent=2))


def _full_tensor(value: torch.Tensor) -> torch.Tensor:
    """Gather a real FSDP tensor outside the differentiable forward path."""
    observed = value.detach()
    observed = observed.full_tensor() if isinstance(observed, DTensor) else observed
    # CPU shards gather through Gloo, then only the detached observation moves
    # to the NPU for comparison with the unsharded reference.
    return observed.to(torch.device("npu", torch.npu.current_device()))


def _save_conditional_checkpoint(reference, destination: Path) -> None:
    """Exercise the actual conditional-to-text namespace through sharded loading."""
    config = Qwen3_5Config.from_dict({
        "text_config": reference.config.to_dict(),
        "vision_config": {
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
            "out_hidden_size": 32, "num_position_embeddings": 16, "deepstack_visual_indexes": [],
        },
    })
    source = Qwen3_5ForConditionalGeneration(config).cpu().float()
    text_state = {
        ("model.language_model." + name.removeprefix("model.") if name.startswith("model.") else name):
        parameter.detach().cpu().clone()
        for name, parameter in reference.named_parameters()
    }
    report = source.load_state_dict(text_state, strict=False)
    assert not report.unexpected_keys, f"Unexpected canonical text keys: {report.unexpected_keys}"
    assert all("visual" in name for name in report.missing_keys), (
        f"Expected only unused vision keys missing, actual={report.missing_keys}"
    )
    source.save_pretrained(destination)


def test_qwen35_fsdp_training() -> None:
    """Match unsharded FP32 gradients, full recomputation and same-topology restore."""
    torch_npu.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    # Match the RL runtime: CPU-offloaded state uses Gloo, NPU compute uses HCCL.
    dist.init_process_group(backend="cpu:gloo,npu:hccl")
    try:
        with tempfile.TemporaryDirectory(prefix="qwen35-fsdp-") as rank_directory:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            torch.manual_seed(7)
            # The real 27B uses 16 key and 48 value heads. Keep its ratio and
            # head dimension while making the tiny head vectors divisible by
            # two, as required by generic DTensor full-state redistribution.
            reference = _build_model(torch.float32, linear_num_key_heads=2)
            ids = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 0, 0]], device="npu")
            path_holder = [rank_directory if rank == 0 else None]
            dist.broadcast_object_list(path_holder, src=0)
            checkpoint_directory = Path(path_holder[0])
            original_path = checkpoint_directory / "original"
            if rank == 0:
                _save_conditional_checkpoint(reference, original_path)
            dist.barrier()
            mesh = MeshContext(dp_size=world_size, dp_shard_size=world_size)
            mesh.build_meshs("npu", world_size)
            results = []
            for offload, recomputation in ((False, None), (False, "full"), (True, "full")):
                setup = DistributedSetup(
                    mesh_context=mesh,
                    strategy_config=FSDP2Config(dp_shard_size=world_size, enable_offload=offload),
                )
                reference.zero_grad(set_to_none=True)
                expected_loss = _loss(reference, ids)
                expected_loss.backward()
                sharded = _build_model(
                    torch.float32, recomputation, distributed_setup=setup, checkpoint_path=str(original_path)
                )
                assert any(isinstance(parameter, DTensor) for parameter in sharded.parameters()), (
                    f"Expected real FSDP parameters, world_size={world_size}"
                )
                # Build the optimizer before any forward: the FSDP root keeps
                # full parameters after inference until a backward reshards it.
                optimizer = torch.optim.SGD(sharded.parameters(), lr=1e-2, foreach=False)
                for name, parameter in sharded.named_parameters():
                    torch.testing.assert_close(
                        _full_tensor(parameter), reference.get_parameter(name), rtol=0, atol=0,
                        msg=f"Conditional-source sharded text extraction mismatch: {name}",
                    )
                actual_loss = _loss(sharded, ids)
                torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-4, atol=2e-5)
                actual_loss.backward()
                for name, parameter in sharded.named_parameters():
                    assert parameter.grad is not None, f"Missing sharded gradient for parameter={name}"
                    actual_gradient = _full_tensor(parameter.grad)
                    expected_gradient = reference.get_parameter(name).grad
                    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-4, atol=2e-5)
                    assert torch.isfinite(actual_gradient).all(), f"Nonfinite FSDP gradient for parameter={name}"
                if offload:
                    assert any(parameter.to_local().device.type == "cpu" for parameter in sharded.parameters()), (
                        f"Expected real CPU-offloaded parameter shards, world_size={world_size}"
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                full_state = get_model_state_dict(
                    sharded, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
                )
                resumed_path = checkpoint_directory / f"{recomputation or 'off'}-cpuoffload-{offload}"
                if rank == 0:
                    # Export the actual gathered weights under the original HF
                    # architecture identity, rather than a dynamic FSDP class name.
                    reference.save_pretrained(resumed_path, state_dict=full_state)
                dist.barrier()
                restored = _build_model(
                    torch.float32, recomputation, distributed_setup=setup, checkpoint_path=str(resumed_path)
                )
                resumed_optimizer = torch.optim.SGD(restored.parameters(), lr=1e-2, foreach=False)
                for name, parameter in sharded.named_parameters():
                    torch.testing.assert_close(
                        _full_tensor(parameter), _full_tensor(restored.get_parameter(name)),
                        rtol=2e-4, atol=2e-5, msg=f"Restored parameter mismatch: {name}",
                    )
                with torch.no_grad():
                    torch.testing.assert_close(_forward(sharded, ids), _forward(restored, ids), rtol=2e-4, atol=2e-5)
                # SGD has no moment state: this isolates model/shard checkpoint
                # recovery. The unsharded worker separately checks AdamW moments.
                for current_model in (sharded, restored):
                    _loss(current_model, ids).backward()
                for current_model, current_optimizer in ((sharded, optimizer), (restored, resumed_optimizer)):
                    assert {id(parameter) for parameter in current_model.parameters()} == {
                        id(parameter) for group in current_optimizer.param_groups for parameter in group["params"]
                    }, "Optimizer must retain the actual sharded Parameters after backward"
                for name, parameter in sharded.named_parameters():
                    torch.testing.assert_close(
                        _full_tensor(parameter.grad), _full_tensor(restored.get_parameter(name).grad),
                        rtol=2e-4, atol=2e-5, msg=f"Restored next-step gradient mismatch: {name}",
                    )
                for current_optimizer in (optimizer, resumed_optimizer):
                    current_optimizer.step()
                for name, parameter in sharded.named_parameters():
                    torch.testing.assert_close(
                        _full_tensor(parameter), _full_tensor(restored.get_parameter(name)),
                        rtol=2e-4, atol=2e-5, msg=f"Restored next-step update mismatch: {name}",
                    )
                results.append({
                    "recomputation": recomputation or "off", "cpu_offload": offload,
                    "loss": actual_loss.detach().item(),
                })
            torch.npu.synchronize()
            result = {
                "world_size": world_size, "rank": rank, "dtype": "float32", "tp": 1, "cp": 1,
                "head_dim": 128, "linear_key_heads": 2, "linear_value_heads": 6, "value_key_head_ratio": 3,
                "conditional_source_text_extract_exact": True,
                "source": "public HyperAutoModel + real HCCL FSDP", "gradient_and_loss_equivalence": True,
                "checkpoint_next_update": True, "optimizer": "stateless SGD", "cases": results,
                "scope": "tiny hybrid FSDP model contract, not full 27B or RL optimizer checkpoint acceptance",
            }
            print(json.dumps(result))
            output = os.environ.get("HP_QWEN35_ST_OUTPUT")
            if output:
                destination = Path(output)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / f"tiny-fsdp-{world_size}-rank{rank}.json").write_text(json.dumps(result, indent=2))
    finally:
        dist.destroy_process_group()
