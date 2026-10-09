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
"""Explicit real Qwen3.8-27B FSDP4 capacity and training system tests."""

import json
import os
import resource
import time
from collections import Counter
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch_npu
from safetensors import safe_open
from transformers import AutoTokenizer

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.build_options import FSDP2Config, FSDP2MixedPrecisionConfig


def _record_stage(stage: str, started: float, **details) -> None:
    """Persist completed stages before proceeding into the next capacity window."""
    result = {
        "stage": stage, "rank": dist.get_rank(), "world_size": dist.get_world_size(),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_npu_allocated_bytes": torch.npu.max_memory_allocated(),
        "peak_npu_reserved_bytes": torch.npu.max_memory_reserved(),
        "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        **details,
    }
    print(json.dumps(result), flush=True)
    output = os.environ.get("HP_QWEN35_ST_OUTPUT")
    if output:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        filename = destination / f"full27b-rank{dist.get_rank()}.jsonl"
        with filename.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _checkpoint_text_shapes(source: Path) -> dict[str, tuple[int, ...]]:
    """Read canonical text shapes without allocating full checkpoint tensors."""
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    by_file = {}
    for name, filename in index.items():
        if name.startswith("model.language_model.") or name == "lm_head.weight":
            target = name.replace("model.language_model.", "model.", 1)
            by_file.setdefault(filename, []).append((name, target))
    shapes = {}
    for filename, names in by_file.items():
        with safe_open(source / filename, framework="pt", device="cpu") as checkpoint:
            for name, target in names:
                assert target not in shapes, f"Duplicate canonical checkpoint text key: {target}"
                shapes[target] = tuple(checkpoint.get_slice(name).get_shape())
    assert len(shapes) == 851, f"Expected 851 complete checkpoint text keys, actual={len(shapes)}"
    return shapes


def _tensor_histograms(tensors) -> dict[str, dict[str, int]]:
    """Count actual local tensor precision and placement without gathering."""
    local = [tensor.to_local() if isinstance(tensor, DTensor) else tensor for tensor in tensors]
    return {
        "dtype": dict(Counter(str(tensor.dtype) for tensor in local)),
        "device": dict(Counter(tensor.device.type for tensor in local)),
    }


def _validate_model_structure(model, parameters, checkpoint_shapes) -> dict:
    """Require the complete checkpoint namespace and actual hybrid decoder."""
    actual_shapes = {name: tuple(parameter.shape) for name, parameter in parameters.items()}
    assert actual_shapes == checkpoint_shapes, "Actual model text keys/shapes must exactly match the checkpoint"
    layers = model.model.layers
    assert len(layers) == 64, f"Expected 64 actual decoder layers, actual={len(layers)}"
    layer_types = []
    for index, layer in enumerate(layers):
        expected_type = "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        expected_attribute = "self_attn" if expected_type == "full_attention" else "linear_attn"
        assert hasattr(layer, expected_attribute), f"Missing real {expected_type} module at layer {index}"
        layer_types.append(expected_type)
    assert Counter(layer_types) == {"linear_attention": 48, "full_attention": 16}, "Expected real 48 GDN/16 full layers"
    return {
        "decoder_layers": len(layers), "layer_types": layer_types,
        "checkpoint_text_shapes": actual_shapes,
        "checkpoint_text_shapes_sha256": sha256(json.dumps(actual_shapes, sort_keys=True).encode()).hexdigest(),
    }


def _observe_layer_call(index: int, calls: dict[int, list[bool]], _module, _inputs) -> None:
    """Observe actual decoder executions, including checkpoint recomputation."""
    calls[index].append(torch.is_grad_enabled())


def _decoder_execution_module(layer):
    """Hook the executed decoder instead of a wrapper bypassed during recomputation."""
    visited = set()
    while id(layer) not in visited:
        visited.add(id(layer))
        wrapped = getattr(layer, "_checkpoint_wrapped_module", None)
        if wrapped is None:
            wrapped = getattr(layer, "_wrapped_module", None)
        if not isinstance(wrapped, torch.nn.Module):
            return layer
        layer = wrapped
    raise ValueError("Unexpected cyclic checkpoint wrapper around the decoder")


def _validate_gradients(parameters, optimizer) -> None:
    """Keep optimizer ownership, full finite gradients and CPU offload strict."""
    assert {id(parameter) for parameter in parameters.values()} == {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }, "Optimizer must still own the actual sharded Parameters after backward"
    for name, parameter in parameters.items():
        assert parameter.grad is not None, f"Missing real 27B gradient: {name}"
        assert torch.isfinite(parameter.grad.to_local()).all(), f"Nonfinite real 27B gradient: {name}"
        assert parameter.to_local().device.type == "cpu", f"Expected CPU-offloaded parameter shard: {name}"
        assert parameter.grad.to_local().device.type == "cpu", f"Expected CPU-offloaded gradient shard: {name}"


def _sentinel_update_diagnostics(before, gradient, updated, learning_rate: float) -> dict:
    """Distinguish an absent update from BF16 rounding without changing the step."""
    before_fp32 = before.float()
    update_fp32 = gradient.float().mul(learning_rate)
    expected_fp32 = before_fp32 - update_fp32
    expected_rounded = expected_fp32.to(before.dtype)
    return {
        "sentinel_dtype": str(before.dtype), "sentinel_elements": before.numel(),
        "sentinel_gradient_max_abs": gradient.float().abs().max().item(),
        "sentinel_fp32_update_max_abs": update_fp32.abs().max().item(),
        "sentinel_expected_fp32_changed_elements": torch.count_nonzero(expected_fp32 != before_fp32).item(),
        "sentinel_expected_rounded_changed_elements": torch.count_nonzero(expected_rounded != before).item(),
        "sentinel_actual_changed_elements": torch.count_nonzero(updated != before).item(),
        "sentinel_actual_max_abs_diff": (updated.float() - before_fp32).abs().max().item(),
        "sentinel_actual_vs_expected_rounded_mismatch_elements": (
            torch.count_nonzero(updated != expected_rounded).item()
        ),
    }


def _run_full_training(backward: bool) -> None:
    """Run the complete local checkpoint without substituting a smaller model."""
    model_path = os.environ.get("HP_QWEN38_MODEL_PATH")
    if not model_path:
        pytest.skip("Set HP_QWEN38_MODEL_PATH to the verified complete local Qwen3.8-27B checkpoint")
    source = Path(model_path)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    text_config = config["text_config"]
    assert text_config["hidden_size"] == 5120 and text_config["num_hidden_layers"] == 64, (
        "This system test requires the real Qwen3.8-27B architecture"
    )
    assert (source / "model.safetensors.index.json").is_file(), "Expected complete indexed HF checkpoint"
    torch_npu.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.set_num_threads(2)
    dist.init_process_group(backend="cpu:gloo,npu:hccl")
    started = time.perf_counter()
    hooks = []
    try:
        world_size = dist.get_world_size()
        assert world_size == 4, f"Real 27B training test requires FSDP4, actual world_size={world_size}"
        torch.npu.reset_peak_memory_stats()
        _record_stage("start", started, model_path=str(source), backward=backward,
                      runtime={name: version(name) for name in ("torch", "torch-npu", "transformers")})
        mesh = MeshContext(dp_size=4, dp_shard_size=4)
        mesh.build_meshs("npu", 4)
        setup = DistributedSetup(
            mesh_context=mesh, strategy_config=FSDP2Config(
                dp_shard_size=4, enable_offload=True,
                mix_precision=FSDP2MixedPrecisionConfig(param_dtype="bfloat16", reduce_dtype="float32"),
            ),
            fp32_main_params=False,
        )
        model = HyperAutoModelForCausalLM.from_pretrained(
            str(source), torch_dtype=torch.bfloat16, attn_implementation="sdpa", force_hf=True,
            distributed_setup=setup, activation_checkpoint="full", local_files_only=True, trust_remote_code=False,
        ).train()
        # FSDP replaces root Parameters during forward; capture actual shards
        # immediately after construction, before either training or inference.
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, foreach=False)
        assert model.config._attn_implementation == "sdpa", "Expected the production HF SDPA training route"
        parameters = dict(model.named_parameters())
        assert len(parameters) == 851, f"Expected 851 real trainable text parameters, actual={len(parameters)}"
        assert all(isinstance(parameter, DTensor) for parameter in parameters.values()), "Expected real FSDP shards"
        assert all(parameter.requires_grad for parameter in parameters.values()), "Expected full text training scope"
        assert not any("visual" in name or "mtp" in name for name in parameters), "Expected text-only trainable model"
        global_elements = sum(parameter.numel() for parameter in parameters.values())
        assert global_elements > 25_000_000_000, f"Expected real 27B weights, actual text elements={global_elements}"
        structure = _validate_model_structure(model, parameters, _checkpoint_text_shapes(source))
        assert global_elements == 26_895_998_464, f"Expected exact full text element count, actual={global_elements}"
        calls = {index: [] for index in range(len(model.model.layers))}
        execution_layers = [_decoder_execution_module(layer) for layer in model.model.layers]
        hooks = [
            layer.register_forward_pre_hook(partial(_observe_layer_call, index, calls))
            for index, layer in enumerate(execution_layers)
        ]
        _record_stage("build", started, parameter_count=len(parameters), global_parameter_elements=global_elements,
                      local_parameter_elements=sum(parameter.to_local().numel() for parameter in parameters.values()),
                      dtype="bfloat16", tp=1, cp=1, cpu_offload=True, full_recompute=True, fp32_master=False,
                      attention_implementation="sdpa", reduce_dtype="float32",
                      model_type=model.config.model_type, text_namespace="model.* and lm_head.*",
                      gradient_checkpointing_enabled=bool(getattr(model, "is_gradient_checkpointing", False)),
                      layer_wrapper_classes=[type(layer).__name__ for layer in model.model.layers],
                      layer_execution_classes=[type(layer).__name__ for layer in execution_layers],
                      parameter_histograms=_tensor_histograms(parameters.values()), **structure)
        tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
        tokens = tokenizer.encode(
            "Write a Python function that sorts a list and explain the complexity. " * 16,
            add_special_tokens=False,
        )[:64]
        assert len(tokens) == 64, f"Expected 64 actual tokenizer tokens, actual={len(tokens)}"
        ids = torch.tensor([tokens], dtype=torch.long, device="npu")
        sentinel_name = "model.layers.0.linear_attn.in_proj_qkv.weight"
        sentinel_before = parameters[sentinel_name].to_local().detach().cpu().clone() if backward else None
        with torch.set_grad_enabled(backward):
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           position_ids=torch.arange(64, device="npu")[None], use_cache=False).logits
            logprobs = logits[:, :-1].float().log_softmax(-1)
            loss = -logprobs.gather(-1, ids[:, 1:, None]).mean()
        assert torch.isfinite(loss), f"Expected finite actual NLL, loss={loss.detach().item()}"
        torch.npu.synchronize()
        forward_counts = {index: len(observed) for index, observed in calls.items()}
        assert all(count > 0 for count in forward_counts.values()), "Expected all 64 real decoder layers to execute"
        _record_stage("forward", started, loss=loss.detach().item(), sequence_length=64, micro_batch=1,
                      layer_forward_counts=forward_counts)
        if not backward:
            _record_stage("forward_probe_pass", started, scope="real 27B build/forward only; no backward acceptance")
            return
        loss.backward()
        torch.npu.synchronize()
        parameters = dict(model.named_parameters())
        _validate_gradients(parameters, optimizer)
        recomputed = [index for index, observed in calls.items() if len(observed) > forward_counts[index]]
        assert len(recomputed) == 64, f"Expected actual recomputation of all 64 layers, actual={recomputed}"
        sentinel = parameters[sentinel_name]
        assert sentinel.grad.to_local().float().abs().sum().item() > 0, "Expected nonzero real GDN projection gradient"
        _record_stage("backward", started, finite_gradient_parameters=len(parameters),
                      parameter_histograms=_tensor_histograms(parameters.values()),
                      gradient_histograms=_tensor_histograms(parameter.grad for parameter in parameters.values()),
                      recomputed_layers=recomputed, layer_execution_grad_modes=calls)
        optimizer.step()
        diagnostics = _sentinel_update_diagnostics(
            sentinel_before, sentinel.grad.to_local(), sentinel.to_local(), optimizer.param_groups[0]["lr"],
        )
        _record_stage("update_observation", started, **diagnostics)
        assert torch.isfinite(sentinel.to_local()).all(), "Expected finite updated real GDN parameters"
        assert not torch.equal(sentinel_before, sentinel.to_local()), "Expected an actual real 27B GDN parameter update"
        _record_stage("update_pass", started, optimizer="stateless SGD on BF16 shards", fp32_master=False,
                      full_checkpoint_saved=False, full_checkpoint_restore_verified=False,
                      scope="real 27B FSDP4 NLL/backward/finite-gradient/update; excludes RL and optimizer resume",
                      **diagnostics)
    finally:
        for handle in hooks:
            handle.remove()
        dist.destroy_process_group()


def test_qwen38_full_forward() -> None:
    """Measure the real model build and forward window before allocating gradients."""
    _run_full_training(backward=False)


def test_qwen38_full_training() -> None:
    """Require real full-model forward, backward, finite gradients and an update."""
    _run_full_training(backward=True)
