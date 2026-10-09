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
"""Real CPU training contracts for the shared Qwen3.5/Qwen3.8 architecture."""

import copy
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest
import torch
from torch.utils.checkpoint import DefaultDeviceType

pytest.importorskip("transformers", minversion="5.5.4")

# CPU contracts must not initialize an optional accelerator while HF resolves
# its attention integrations. The actual model forward and backward stay real.
with patch("transformers.utils.is_torch_npu_available", return_value=False), patch(
    "transformers.utils.import_utils.is_torch_npu_available", return_value=False
):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig
    from hyper_parallel.models import HyperAutoModelForCausalLM


def _tiny_text_config() -> Qwen3_5TextConfig:
    """Build a small hybrid decoder with both GDN and full attention."""
    return Qwen3_5TextConfig.from_dict({
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "pad_token_id": 0,
        "max_position_embeddings": 128,
        "use_cache": False,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        },
    })


def _build_text_model(activation_checkpoint: Optional[str] = None):
    """Use the public builder while explicitly keeping this contract on CPU."""
    with patch("hyper_parallel.models.build_options.IS_NPU_AVAILABLE", False), patch(
        "hyper_parallel.models.build_options.IS_CUDA_AVAILABLE", False
    ):
        return HyperAutoModelForCausalLM.from_config(
            _tiny_text_config(), torch_dtype=torch.float32, attn_implementation="eager",
            activation_checkpoint=activation_checkpoint,
        ).cpu()


def _forward(model, input_ids: torch.Tensor):
    """Use contiguous valid-token positions for either padding direction."""
    mask = input_ids.ne(0)
    positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
    return model(
        input_ids=input_ids,
        attention_mask=mask.long(),
        position_ids=positions,
        use_cache=False,
    ).logits


def _loss(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Exclude transitions into or out of padding from a next-token loss."""
    logits = _forward(model, input_ids)
    selected = logits[:, :-1].float().log_softmax(-1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    valid = input_ids[:, :-1].ne(0) & input_ids[:, 1:].ne(0)
    return -selected[valid].mean()


class TestQwen35TrainingContracts(unittest.TestCase):
    """Exercise native HF math through HyperParallel's public CPU builder."""

    @classmethod
    def setUpClass(cls) -> None:
        """Bound CPU work without changing the runtime's accelerator policy."""
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore the caller's CPU thread budget."""
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        """Keep model initialization deterministic and isolate CPU RNG changes."""
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()
        self.addCleanup(self.rng.__exit__, None, None, None)
        torch.manual_seed(7)
        self.ids = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 0, 0]])

    def test_full_checkpoint_extracts_exact_trainable_text_weights(self):
        """The multimodal checkpoint loads its actual text names and values."""
        config = Qwen3_5Config.from_dict({
            "text_config": _tiny_text_config().to_dict(),
            "vision_config": {
                "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                "out_hidden_size": 32, "num_position_embeddings": 16, "deepstack_visual_indexes": [],
            },
        })
        source = Qwen3_5ForConditionalGeneration(config).float().eval()
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            with patch("hyper_parallel.models.build_options.IS_NPU_AVAILABLE", False), patch(
                "hyper_parallel.models.build_options.IS_CUDA_AVAILABLE", False
            ):
                model = HyperAutoModelForCausalLM.from_pretrained(
                    directory, torch_dtype=torch.float32, attn_implementation="eager",
                    force_hf=True, local_files_only=True, trust_remote_code=False,
                ).cpu().eval()
        self.assertEqual(type(model).__name__, "Qwen3_5ForCausalLM")
        source_weights = dict(source.named_parameters())
        for name, parameter in model.named_parameters():
            canonical = "model.language_model." + name.removeprefix("model.") if name.startswith("model.") else name
            self.assertIn(canonical, source_weights)
            torch.testing.assert_close(parameter, source_weights[canonical], rtol=0, atol=0)
        self.assertFalse(any("visual" in name or name.startswith("mtp.") for name, _ in model.named_parameters()))
        torch.testing.assert_close(_forward(model, self.ids), _forward(source, self.ids), rtol=2e-5, atol=2e-5)

    def test_gdn_backward_updates_real_parameters(self):
        """Every trainable parameter has finite gradients and GDN changes."""
        model = _build_text_model().train()
        before = model.model.layers[0].linear_attn.in_proj_qkv.weight.detach().clone()
        loss = _loss(model, self.ids)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=f"Missing gradient for parameter={name}")
            self.assertTrue(torch.isfinite(parameter.grad).all(), msg=f"Nonfinite gradient for parameter={name}")
        for name in ("A_log", "dt_bias", "conv1d.weight", "in_proj_qkv.weight", "out_proj.weight"):
            gradient = model.model.layers[0].linear_attn.get_parameter(name).grad
            self.assertGreater(gradient.abs().sum().item(), 0.0, msg=f"Zero GDN gradient for parameter={name}")
        torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False, fused=False).step()
        self.assertFalse(torch.equal(before, model.model.layers[0].linear_attn.in_proj_qkv.weight))

    def test_padding_and_batch_isolation_match_independent_sequences(self):
        """Both padding directions preserve valid outputs and masked gradients."""
        baseline = _build_text_model().train()
        short = self.ids[1:2, :4]
        independent_logits = _forward(baseline, short).detach()
        for padded in (self.ids, torch.tensor([[2, 3, 4, 5, 6, 7], [0, 0, 8, 9, 10, 11]])):
            with self.subTest(padding=padded[1].tolist()):
                batched = copy.deepcopy(baseline)
                independent = copy.deepcopy(baseline)
                valid = padded[1].ne(0)
                torch.testing.assert_close(
                    _forward(batched, padded)[1, valid], independent_logits[0], rtol=2e-5, atol=2e-5
                )
                batch_loss = _loss(batched, padded)
                separate_loss = (_loss(independent, self.ids[:1]) * 5 + _loss(independent, short) * 3) / 8
                torch.testing.assert_close(batch_loss, separate_loss, rtol=2e-5, atol=2e-5)
                batch_loss.backward()
                separate_loss.backward()
                for actual, expected in zip(batched.parameters(), independent.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=3e-5)
                _forward(batched, self.ids[:1])
                torch.testing.assert_close(_forward(batched, short), independent_logits, rtol=0, atol=0)

    def test_activation_recomputation_preserves_logits_and_gradients(self):
        """Checkpointing really reruns GDN and retains the same derivative."""
        default_device = DefaultDeviceType.get_device_type()
        # Checkpoint otherwise probes the registered NPU even for CPU inputs.
        DefaultDeviceType.set_device_type("cpu")
        self.addCleanup(DefaultDeviceType.set_device_type, default_device)
        plain = _build_text_model().train()
        recomputed = _build_text_model(activation_checkpoint="full").train()
        recomputed.load_state_dict(plain.state_dict())
        calls = []
        handle = recomputed.model.layers[0].register_forward_pre_hook(lambda module, inputs: calls.append(inputs))
        self.addCleanup(handle.remove)
        expected_loss = _loss(plain, self.ids)
        actual_loss = _loss(recomputed, self.ids)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        expected_loss.backward()
        actual_loss.backward()
        self.assertGreaterEqual(len(calls), 2, msg=f"Expected GDN recomputation, observed forward calls={len(calls)}")
        for actual, expected in zip(recomputed.parameters(), plain.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-5)

    def test_checkpoint_restores_optimizer_and_next_update(self):
        """HF text export and optimizer state reproduce the following step."""
        model = _build_text_model().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False, fused=False)
        _loss(model, self.ids).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            state_path = Path(directory) / "optimizer.pt"
            torch.save(optimizer.state_dict(), state_path)
            with patch("hyper_parallel.models.build_options.IS_NPU_AVAILABLE", False), patch(
                "hyper_parallel.models.build_options.IS_CUDA_AVAILABLE", False
            ):
                restored = HyperAutoModelForCausalLM.from_pretrained(
                    directory, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True,
                ).cpu().train()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, foreach=False, fused=False)
            restored_optimizer.load_state_dict(torch.load(state_path, weights_only=True))
        torch.testing.assert_close(_forward(model, self.ids), _forward(restored, self.ids), rtol=0, atol=0)
        for current_model, current_optimizer in ((model, optimizer), (restored, restored_optimizer)):
            _loss(current_model, self.ids).backward()
            current_optimizer.step()
        self.assertEqual(tuple(model.state_dict()), tuple(restored.state_dict()))
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, restored.get_parameter(name), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
