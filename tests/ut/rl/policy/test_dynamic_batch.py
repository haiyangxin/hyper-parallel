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
"""CPU contracts for dynamic Actor scheduling and unchanged optimizer objectives."""
from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pytest
import torch

from rl.algorithm import build_algorithm
from rl.dataset.contracts import ExperienceBatch
from rl.roles.policy.actor import Actor
from rl.roles.policy.dynamic_batch import equalize_batch_count, plan_token_batches, select_experience_rows


class TokenPolicy(torch.nn.Module):
    """Small differentiable policy whose outputs are independent of batch padding."""

    supports_token_log_probs = True
    supports_token_log_probs_mask = True

    def __init__(self) -> None:
        """Initialize two trainable scalars."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.03))
        self.bias = torch.nn.Parameter(torch.tensor(-0.7))

    def forward(self, input_ids: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        """Return differentiable next-token scores."""
        return self.weight * input_ids[:, 1:].float() + self.bias


def make_experience(lengths: list[int]) -> ExperienceBatch:
    """Create variable-length rows with distinct tokens, targets and identities."""
    width = max(lengths)
    sequences = torch.arange(len(lengths) * width).reshape(len(lengths), width) % 13
    attention = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]
    actions = attention.clone()
    actions[:, :2] = False
    old = 0.03 * sequences[:, 1:].float() - 0.7
    return ExperienceBatch(
        trajectories=tuple(SimpleNamespace(token_ids=sequences[index, :length], metadata={}, worker_policy_version=None)
                           for index, length in enumerate(lengths)),
        sequences=sequences, attention_mask=attention, action_mask=actions,
        rewards=torch.arange(len(lengths)).float(), old_log_probs=old,
        responses=tuple(str(index) for index in range(len(lengths))), generation_seconds=0.0,
        advantages=torch.sin(sequences[:, 1:].float()), reference_log_probs=old - 0.1,
    )


@pytest.mark.parametrize("lengths,budget", [([8, 3, 7, 4, 6, 2], 16), ([9, 8, 2, 2], 12), ([4], 4)])
def test_plan_covers_every_row_with_padded_budget(lengths, budget):
    """Selection neither drops samples nor mistakes useful tokens for padded allocation."""
    groups = plan_token_batches(lengths, budget)
    assert sorted(index for group in groups for index in group) == list(range(len(lengths)))
    assert all(len(group) * max(lengths[index] for index in group) <= budget for group in groups)
    assert plan_token_batches(lengths, budget) == groups


@pytest.mark.parametrize("lengths,budget", [([9], 8), ([1], 8), ([], 8), ([2], 0)])
def test_plan_rejects_oversized_or_empty_sequences(lengths, budget):
    """An impossible budget fails explicitly instead of silently exceeding its limit."""
    with pytest.raises(ValueError):
        plan_token_batches(lengths, budget)


def test_equalization_splits_without_duplicates_or_empty_batches():
    """DP call matching must preserve samples and already-established memory bounds."""
    original = [[0, 1, 2], [3, 4]]
    groups = equalize_batch_count(original, 4)
    assert len(groups) == 4
    assert all(groups)
    assert sorted(index for group in groups for index in group) == list(range(5))
    assert original == [[0, 1, 2], [3, 4]]
    with pytest.raises(ValueError, match="empty"):
        equalize_batch_count(original, 6)


def test_padding_overflow_candidates_are_reused_in_later_groups():
    """Short first-fit fillers must not turn into a separate tiny batch for every long group."""
    groups = plan_token_batches([10, 10, 10, 10, 6, 6], 26)
    assert groups == [[0, 1], [2, 3], [4, 5]]


def test_selection_preserves_all_tensor_targets_and_row_identities():
    """Row permutations carry teacher, value and policy targets together without mutation."""
    experience = make_experience([8, 3, 7, 4])
    experience = replace(experience, returns=experience.advantages + 1, values=experience.advantages - 1,
                         rollout_log_probs=experience.old_log_probs, teacher_log_probs=experience.old_log_probs,
                         teacher_keys=("a", "b", "c", "d"), bootstrap_values=experience.rewards + 2)
    order = [2, 0, 3, 1]
    selected = select_experience_rows(experience, order)
    for field in fields(experience):
        value = getattr(experience, field.name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(getattr(selected, field.name), value[order])
    assert selected.responses == ("2", "0", "3", "1")
    assert selected.teacher_keys == ("c", "a", "d", "b")
    assert selected.trajectories[0] is experience.trajectories[2]
    assert experience.responses == ("0", "1", "2", "3")


def make_actor(budget: int | None = None, mini_batch: int = 4, epochs: int = 1) -> Actor:
    """Build a CPU Actor sharing the production loss and update loop."""
    model = TokenPolicy()
    return Actor(model, build_algorithm({"name": "grpo", "loss_aggregation": "token-mean", "kl_coef": 0.01}),
                 2, optimizer=torch.optim.SGD(model.parameters(), lr=0.01), device=torch.device("cpu"),
                 response_mini_batch_size=mini_batch, update_epochs=epochs, max_tokens_per_micro_batch=budget)


@pytest.mark.parametrize("budget", [8, 16, 32])
def test_dynamic_update_matches_fixed_batch_objective(budget):
    """Variable sizes and ordering preserve gradients across optimizer steps and epochs."""
    experience = make_experience([5, 8, 3, 7, 4, 6, 8, 3])
    fixed = make_actor(epochs=2)
    dynamic = make_actor(budget, epochs=2)
    with patch("rl.roles.policy.actor.hsdp_sync_stream"), patch(
        "rl.roles.policy.actor.clip_grad_norm_", torch.nn.utils.clip_grad_norm_,
    ):
        fixed_metrics = fixed.update(experience)
        dynamic_metrics = dynamic.update(experience)
    for expected, actual in zip(fixed.parameters(), dynamic.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert dynamic_metrics.optimizer_steps == fixed_metrics.optimizer_steps == 4
    assert dynamic_metrics.valid_tokens == fixed_metrics.valid_tokens
    assert dynamic_metrics.total_loss == pytest.approx(fixed_metrics.total_loss, abs=1e-7)
    assert dynamic_metrics.input_tokens == 2 * sum([5, 8, 3, 7, 4, 6, 8, 3])
    assert dynamic_metrics.padded_input_tokens >= dynamic_metrics.input_tokens
    assert dynamic_metrics.micro_batches >= 4


def test_dp_micro_batch_counts_are_equalized_before_backward():
    """A rank with shorter sequences splits its groups to match its peer's HSDP call count."""
    actor = make_actor(32)
    actor._dp_size = 2
    actor._dp_group_info = SimpleNamespace(group="dp")

    def gather(output: list, local: list, group: str) -> None:
        """Supply a peer's micro-batch count without a process group."""
        assert group == "dp"
        output[:] = [local, [(3, 4)]]

    with patch("rl.roles.policy.actor.dist.all_gather_object", side_effect=gather):
        groups = actor._plan_dynamic_update(make_experience([3, 4, 3, 4]))[0]
    assert len(groups) == 3
    assert sorted(index for group in groups for index in group) == [0, 1, 2, 3]


def test_dynamic_schedule_keeps_optimizer_minibatch_membership():
    """Reordering stays within each original optimizer mini-batch."""
    actor = make_actor(16)
    plans = actor._plan_dynamic_update(make_experience([3, 8, 3, 7, 4, 6, 8, 3]))
    assert sorted(index for group in plans[0] for index in group) == [0, 1, 2, 3]
    assert sorted(index for group in plans[1] for index in group) == [4, 5, 6, 7]


def test_dynamic_preserves_partial_final_optimizer_minibatch():
    """The final shorter mini-batch still performs exactly one complete update."""
    experience = make_experience([5, 8, 3, 7, 4, 6])
    fixed, dynamic = make_actor(), make_actor(16)
    with patch("rl.roles.policy.actor.hsdp_sync_stream"), patch(
        "rl.roles.policy.actor.clip_grad_norm_", torch.nn.utils.clip_grad_norm_,
    ):
        fixed.update(experience)
        metrics = dynamic.update(experience)
    assert metrics.optimizer_steps == 2
    for expected, actual in zip(fixed.parameters(), dynamic.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_dynamic_rejects_noncanonical_padding_before_model_forward():
    """No token may be trimmed using metadata that disagrees with the actual input mask."""
    experience = make_experience([4, 8, 3, 6])
    experience.attention_mask[0, 0] = False
    actor = make_actor(16)
    with pytest.raises(ValueError, match="right-padded"):
        actor.update(experience)


@pytest.mark.parametrize("budget", [True, 0, -1, 2.5])
def test_actor_rejects_invalid_token_budget(budget):
    """Malformed budgets fail before optimization."""
    with pytest.raises(ValueError, match="positive integer"):
        make_actor(budget)


class LogitsPolicy(TokenPolicy):
    """Standard HF logits contract used by the current dense Qwen3 recipe."""

    supports_token_log_probs = False
    supports_token_log_probs_mask = False

    def __init__(self) -> None:
        """Record rectangular inputs for the real logits execution branch."""
        super().__init__()
        self.input_shapes = []

    def forward(self, input_ids: torch.Tensor, **_kwargs: object) -> dict[str, torch.Tensor]:
        """Return full vocabulary logits without selected-probability extensions."""
        self.input_shapes.append(tuple(input_ids.shape))
        vocabulary = torch.arange(13).float()
        features = input_ids.float().unsqueeze(-1) * vocabulary / 13
        return {"logits": self.weight * features + self.bias * vocabulary}


def test_standard_logits_dynamic_update_trims_only_training_and_preserves_gradients():
    """Dense HF logits need no selected-probability API to support token-budget updates."""
    experience = make_experience([3, 8, 4, 6])
    models = [LogitsPolicy(), LogitsPolicy()]
    actors = [Actor(model, build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}), 2,
                    optimizer=torch.optim.SGD(model.parameters(), lr=0.01), device=torch.device("cpu"),
                    response_mini_batch_size=4, max_tokens_per_micro_batch=budget)
              for model, budget in zip(models, [None, 12])]
    old = actors[0].compute_log_probs(experience)
    dynamic_old = actors[1].compute_log_probs(experience)
    torch.testing.assert_close(dynamic_old, old, rtol=0, atol=0)
    assert models[1].input_shapes == [(2, 8), (2, 8)]
    experience = replace(experience, old_log_probs=old, reference_log_probs=old)
    for model in models:
        model.input_shapes.clear()
    with patch("rl.roles.policy.actor.hsdp_sync_stream"), patch(
        "rl.roles.policy.actor.clip_grad_norm_", torch.nn.utils.clip_grad_norm_,
    ):
        fixed, dynamic = [actor.update(experience) for actor in actors]
    assert models[0].input_shapes == [(2, 8), (2, 8)]
    assert all(rows * width <= 12 for rows, width in models[1].input_shapes)
    assert dynamic.padded_input_tokens < 32
    assert dynamic.optimizer_steps == fixed.optimizer_steps == 1
    for expected, actual in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


class CausalLogitsPolicy(torch.nn.Module):
    """Sequence-coupled CPU model to test causal context and padding semantics."""

    def __init__(self) -> None:
        """Build attention and a vocabulary projection without accelerator dependencies."""
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 8)
        self.attention = torch.nn.MultiheadAttention(8, 2, batch_first=True)
        self.projection = torch.nn.Linear(8, 13)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Attend only to the same row's valid causal prefix."""
        hidden = self.embedding(input_ids)
        causal = torch.ones(input_ids.shape[1], input_ids.shape[1], dtype=torch.bool).triu(1)
        hidden, _ = self.attention(hidden, hidden, hidden, attn_mask=causal,
                                   key_padding_mask=~attention_mask.bool(), need_weights=False)
        return {"logits": self.projection(hidden)}


def test_dynamic_update_preserves_causal_attention_context_and_loss():
    """Trimming and regrouping keep the complete context and update of a sequence-coupled model."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        models = [CausalLogitsPolicy(), CausalLogitsPolicy()]
    models[1].load_state_dict(models[0].state_dict())
    actors = [Actor(model, build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}), 2,
                    optimizer=torch.optim.SGD(model.parameters(), lr=0.01), device=torch.device("cpu"),
                    response_mini_batch_size=6, update_epochs=2, max_tokens_per_micro_batch=budget)
              for model, budget in zip(models, [None, 24])]
    experience = make_experience([3, 8, 4, 6, 8, 4])
    old = actors[0].compute_log_probs(experience)
    experience = replace(experience, old_log_probs=old, reference_log_probs=old)
    with patch("rl.roles.policy.actor.hsdp_sync_stream"), patch(
        "rl.roles.policy.actor.clip_grad_norm_", torch.nn.utils.clip_grad_norm_,
    ):
        fixed, dynamic = [actor.update(experience) for actor in actors]
    assert dynamic.optimizer_steps == fixed.optimizer_steps == 2
    assert dynamic.total_loss == pytest.approx(fixed.total_loss, abs=2e-7)
    for expected, actual in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-7)


class TestDynamicLogProbForward(TestCase):
    """Frozen forwards retain causal values and align FSDP calls without optimizer inputs."""

    def test_causal_probabilities_restore_rows_and_training_mode(self) -> None:
        """Length sorting must preserve every valid next-token probability and model state."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            model = CausalLogitsPolicy()
        actor = Actor(model, build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}), 2,
                      optimizer=torch.optim.SGD(model.parameters(), lr=0.01), device=torch.device("cpu"))
        experience = replace(make_experience([3, 8, 4, 6, 8, 4]), old_log_probs=None,
                             reference_log_probs=None, advantages=None)
        expected = actor.compute_log_probs(experience)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        actor._log_prob_max_tokens_per_micro_batch = 16
        actual = actor.compute_log_probs(experience)
        mask = experience.attention_mask[:, 1:].bool()
        torch.testing.assert_close(actual[mask], expected[mask], atol=1e-6, rtol=1e-6)
        self.assertEqual(actual.shape, expected.shape)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(actor.training)
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
            self.assertIsNone(value.grad)

    def test_reference_aligns_calls_and_respects_rectangular_budget(self) -> None:
        """Frozen Reference shares the DP group and splits batches without duplicating rows."""
        model = LogitsPolicy()
        actor = Actor(model, build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}), 2,
                      dp_size=2, dp_group_info=SimpleNamespace(group="dp"),
                      log_prob_max_tokens_per_micro_batch=32)

        def gather(output: list, local: tuple, group: str) -> None:
            """Supply peer call counts without initializing distributed hardware."""
            self.assertEqual(group, "dp")
            output[:] = [local, (3, 4)]

        with patch("rl.roles.policy.actor.dist.all_gather_object", side_effect=gather):
            result = actor.compute_log_probs(make_experience([3, 4, 3, 4]))
        self.assertEqual(len(model.input_shapes), 3)
        self.assertEqual(sum(rows for rows, _ in model.input_shapes), 4)
        self.assertTrue(all(rows * width <= 32 for rows, width in model.input_shapes))
        self.assertEqual(result.shape, (4, 3))
        self.assertFalse(actor.training)

    def test_invalid_padding_and_budget_fail_before_forward(self) -> None:
        """Never silently truncate context or enter collectives with invalid metadata."""
        for budget in (True, 0, -1, 2.5):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "positive integer"):
                Actor(LogitsPolicy(), build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}), 2,
                      log_prob_max_tokens_per_micro_batch=budget)
        model = LogitsPolicy()
        actor = Actor(model, build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}), 2,
                      log_prob_max_tokens_per_micro_batch=8)
        experience = make_experience([3, 8])
        experience.attention_mask[0, 0] = False
        with self.assertRaisesRegex(ValueError, "right-padded"):
            actor.compute_log_probs(experience)
        self.assertFalse(model.input_shapes)
        self.assertFalse(actor.training)

    def test_impossible_peer_count_fails_before_forward(self) -> None:
        """A peer needing more calls than local rows cannot be padded with duplicate samples."""
        model = LogitsPolicy()
        actor = Actor(model, build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}), 2, dp_size=2,
                      dp_group_info=SimpleNamespace(group="dp"), log_prob_max_tokens_per_micro_batch=16)

        def gather(output: list, local: tuple, group: str) -> None:
            """Supply peer call counts without initializing distributed hardware."""
            output[:] = [local, (5, 6)]

        with patch("rl.roles.policy.actor.dist.all_gather_object", side_effect=gather):
            with self.assertRaisesRegex(ValueError, "empty batches"):
                actor.compute_log_probs(make_experience([3, 4]))
        self.assertFalse(model.input_shapes)
