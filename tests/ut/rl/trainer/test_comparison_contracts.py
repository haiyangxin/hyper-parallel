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
"""CPU contracts for comparison policy anchors, scoring and final evaluation."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rl.dataset.contracts import ExperienceBatch
from rl.trainer import SyncTrainer

from examples.gsm8k.agent import compute_gsm8k_reward
from examples.gsm8k.verl_reward import StrictComparisonGSM8KEnvironment, compute_verl_reward


class TestComparisonContracts(unittest.TestCase):
    """Keep sampled evidence intact while optionally recomputing the update anchor."""

    def test_actor_anchor_without_sampling_probabilities(self) -> None:
        """A recomputed anchor must work without inventing rollout diagnostics."""
        rollout = ExperienceBatch(
            trajectories=(), sequences=torch.ones(2, 4, dtype=torch.long),
            attention_mask=torch.ones(2, 4), action_mask=torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]]),
            rewards=torch.tensor([1.0, 0.0]), old_log_probs=None,
            rollout_log_probs=None, responses=("", ""), generation_seconds=0.0,
        )
        recomputed = torch.full((2, 3), -1.5, requires_grad=True)
        trainer = SyncTrainer.__new__(SyncTrainer)
        trainer._old_log_prob_source = "actor"
        trainer._consistency_profile = "off"
        trainer.algorithm = SimpleNamespace(
            name="grpo", requirements=SimpleNamespace(data=SimpleNamespace(reference_log_probs=False, values=False)),
        )
        trainer.actor = Mock()
        trainer.actor.compute_log_probs.return_value = recomputed
        trainer._run_rank_synchronized = lambda _operation, callback: callback()
        trainer.experience_preparer = Mock()
        trainer.experience_preparer.prepare.side_effect = lambda batch, **_kwargs: batch
        with patch("rl.trainer.summarize_training_diagnostics",
                   return_value={"training/total_tokens": 8.0}) as diagnostics:
            prepared, metrics = trainer._prepare_experience(rollout, True, {})
        torch.testing.assert_close(prepared.old_log_probs, recomputed.detach())
        self.assertFalse(prepared.old_log_probs.requires_grad)
        self.assertIsNone(prepared.rollout_log_probs)
        self.assertIsNone(rollout.old_log_probs)
        self.assertEqual(metrics, {"training/total_tokens": 8.0})
        diagnostics.assert_called_once_with(prepared, None)

    def test_probability_source_and_diagnostics_are_independent(self) -> None:
        """Diagnostics alone must not change anchors; actor selection forces inference."""
        for source, diagnostics in (("rollout", False), ("rollout", True), ("actor", False)):
            with self.subTest(source=source, diagnostics=diagnostics):
                sampled = torch.full((2, 3), -2.0)
                rollout = ExperienceBatch(
                    trajectories=(), sequences=torch.ones(2, 4, dtype=torch.long),
                    attention_mask=torch.ones(2, 4), action_mask=torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]]),
                    rewards=torch.tensor([1.0, 0.0]), old_log_probs=sampled,
                    rollout_log_probs=sampled.clone(), responses=("", ""), generation_seconds=0.0,
                )
                recomputed = torch.full((2, 3), -1.5, requires_grad=True)
                trainer = SyncTrainer.__new__(SyncTrainer)
                trainer._old_log_prob_source = source
                trainer._consistency_profile = "off"
                trainer.algorithm = SimpleNamespace(
                    name="grpo",
                    requirements=SimpleNamespace(data=SimpleNamespace(reference_log_probs=False, values=False)),
                )
                trainer.actor = Mock()
                trainer.actor.compute_log_probs.return_value = recomputed
                trainer._run_rank_synchronized = lambda _operation, callback: callback()
                trainer.experience_preparer = Mock()
                trainer.experience_preparer.prepare.side_effect = lambda batch, **_kwargs: batch
                timings = {}
                with patch("rl.trainer.summarize_training_diagnostics", return_value={}) as summary:
                    prepared, _ = trainer._prepare_experience(rollout, diagnostics, timings)
                self.assertEqual(summary.call_count, int(diagnostics))
                expected = recomputed.detach() if source == "actor" else sampled
                torch.testing.assert_close(prepared.old_log_probs, expected)
                torch.testing.assert_close(prepared.rollout_log_probs, sampled)
                torch.testing.assert_close(rollout.old_log_probs, sampled)
                self.assertFalse(prepared.old_log_probs.requires_grad)
                self.assertEqual(trainer.actor.compute_log_probs.call_count, int(diagnostics or source == "actor"))
                self.assertEqual("old_log_prob" in timings, diagnostics or source == "actor")

    def test_final_evaluation_does_not_require_checkpoint_saving(self) -> None:
        """An explicit end-of-run evaluation runs once even when saving is disabled."""
        trainer = SyncTrainer.__new__(SyncTrainer)
        trainer.resolved_config = {
            "train": {"checkpoint": {"save_final": False}},
            "evaluation": {"at_end": True},
        }
        trainer.state = SimpleNamespace(global_step=1, max_steps=1)
        trainer.checkpoints = Mock()
        trainer.rollout_engine = SimpleNamespace(policy_version=1)
        trainer._release_training_state_for_rollout = Mock()
        trainer.train_dataloader = []
        trainer.evaluator = Mock(last_step=-1)
        trainer.evaluator.run.return_value = ({"validation/accuracy": 0.5}, [])
        trainer._tracker = Mock()
        self.assertTrue(trainer._train_and_checkpoint())
        trainer.evaluator.run.assert_called_once_with(1)
        trainer._tracker.log.assert_called_once()

    def test_evaluation_schedule_without_checkpoint_and_without_final_duplicate(self) -> None:
        """Evaluate the initial and published final policy once with saving disabled."""
        trainer = SyncTrainer.__new__(SyncTrainer)
        trainer.resolved_config = {
            "train": {"checkpoint": {"save_final": False}},
            "evaluation": {"before_train": True, "every_n_steps": 5, "at_end": True},
        }
        trainer.state = SimpleNamespace(global_step=0, max_steps=5)
        trainer.checkpoints = Mock()
        trainer.checkpoints.will_save.return_value = False
        trainer.rollout_engine = SimpleNamespace(policy_version=0)
        trainer._release_training_state_for_rollout = Mock()
        trainer.train_dataloader = [object()] * 5
        trainer.evaluator = Mock(last_step=-1)
        calls = []

        def evaluate(step: int) -> tuple[dict[str, float], list]:
            """Record the exact published policy version."""
            calls.append(step)
            self.assertEqual(trainer.state.global_step, step)
            trainer.evaluator.last_step = step
            return {"validation/accuracy": 0.5}, []

        def train_step(_batch: object) -> None:
            """Advance a simulated successful publication."""
            trainer.state.global_step += 1
            if trainer._evaluation_due(trainer.state.global_step):
                trainer.evaluator.run(trainer.state.global_step)

        trainer.evaluator.run.side_effect = evaluate
        trainer._tracker = Mock()
        trainer._next_batch = lambda iterator: (next(iterator), iterator)
        trainer._train_step = train_step
        self.assertTrue(trainer._train_and_checkpoint())
        self.assertEqual(calls, [0, 5])
        trainer.checkpoints.will_save.assert_not_called()

    def test_periodic_evaluation_preserves_legacy_schedule_when_unspecified(self) -> None:
        """An explicit interval overrides saving; omitted intervals retain existing behavior."""
        trainer = SyncTrainer.__new__(SyncTrainer)
        trainer.evaluator = Mock()
        trainer.checkpoints = Mock()
        trainer.checkpoints.will_save.return_value = True
        for interval, expected in ((None, True), (0, False), (5, True), (3, False)):
            with self.subTest(interval=interval):
                trainer.resolved_config = {"evaluation": {"every_n_steps": interval}}
                self.assertEqual(trainer._evaluation_due(5), expected)
                self.assertFalse(trainer._evaluation_due(0))
        trainer.evaluator = None
        self.assertFalse(trainer._evaluation_due(5))

    def test_strict_environment_uses_baseline_training_reward(self) -> None:
        """Opting into strict scoring changes training reward without changing the numeric scorer."""
        prompt = SimpleNamespace(ground_truth="42")
        self.assertEqual(StrictComparisonGSM8KEnvironment.score_reward(prompt, answer="42"), 0.0)
        self.assertEqual(StrictComparisonGSM8KEnvironment.score_reward(prompt, answer="#### 42"), 1.0)
        self.assertEqual(compute_gsm8k_reward("42", "42"), 1.0)

    def test_numeric_scoring_and_baseline_scoring_are_both_retained(self) -> None:
        """Numeric normalization and strict format scoring have intentional differences."""
        cases = [
            ("42", "42", 1.0, 0.0),
            ("#### 42.0", "42", 1.0, 0.0),
            ("#### 42 therefore", "42", 0.0, 1.0),
            ("#### 1,234", "1234", 1.0, 1.0),
            ("#### 41\n#### 42", "42", 1.0, 1.0),
            ("#### 42" + "x" * 301, "42", 0.0, 0.0),
        ]
        for answer, target, numeric, strict in cases:
            with self.subTest(answer=answer):
                self.assertEqual(compute_gsm8k_reward(answer, target), numeric)
                self.assertEqual(compute_verl_reward(answer, target), strict)


class TestResidentOptimizer(unittest.TestCase):
    """Check residency at publication without real accelerator allocation."""

    def test_residency_allows_scalar_cpu_counter_but_rejects_moment_offload(self) -> None:
        """Tensor moments and master parameters follow the selected device."""
        parameter = torch.nn.Parameter(torch.empty(2, device="meta"))
        optimizer = SimpleNamespace(param_groups=[{"params": [parameter]}], state={parameter: {
            "step": torch.tensor(1.0), "exp_avg": torch.empty(2, device="meta"),
        }})
        SyncTrainer._validate_optimizer_device_residency(optimizer, torch.device("meta"))
        optimizer.state[parameter]["exp_avg"] = torch.zeros(2)
        with self.assertRaisesRegex(RuntimeError, "optimizer state"):
            SyncTrainer._validate_optimizer_device_residency(optimizer, torch.device("meta"))
        optimizer.state[parameter]["exp_avg"] = torch.zeros(())
        with self.assertRaisesRegex(RuntimeError, "optimizer state"):
            SyncTrainer._validate_optimizer_device_residency(optimizer, torch.device("meta"))
        optimizer.param_groups[0]["params"] = [torch.zeros(2)]
        with self.assertRaisesRegex(RuntimeError, "optimizer parameter"):
            SyncTrainer._validate_optimizer_device_residency(optimizer, torch.device("meta"))
