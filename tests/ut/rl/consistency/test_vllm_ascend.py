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
"""Partial-prefill RNG correctness and cleanup for reviewed Ascend runners."""

from types import SimpleNamespace
from typing import Any
import unittest

from rl.consistency.vllm_ascend import patch_partial_prefill_rng


class OffsetGenerator:
    """Expose the NPU generator offset interface without accelerator allocation."""

    def __init__(self, offset: int) -> None:
        """Initialize an observable generator position."""
        self.offset = offset

    def get_offset(self) -> int:
        """Return the current seeded position."""
        return self.offset

    def set_offset(self, offset: int) -> None:
        """Restore a captured seeded position."""
        self.offset = offset


class TestPartialPrefillRNG(unittest.TestCase):
    """Keep valid samples while rolling discarded rows back to their actual offsets."""

    def _runner(self) -> Any:
        """Build the new runner call signature with its upstream fixed-four rewind."""
        class Runner:
            """Model the 0.23 sample/bookkeeping interaction on CPU."""

            def __init__(self) -> None:
                """Initialize two request rows with independent seeded positions."""
                self.input_batch = SimpleNamespace(generators={0: OffsetGenerator(32), 1: OffsetGenerator(64)})
                self.discard_request_indices = SimpleNamespace(np=[1])
                self.num_discarded_requests = 1
                self.sample_failure = self.bookkeeping_failure = False

            def _sample(self, logits: Any, spec_decode_metadata: Any) -> Any:
                for generator in self.input_batch.generators.values():
                    generator.set_offset(generator.get_offset() + 12)
                if self.sample_failure:
                    raise RuntimeError("sample failed")
                return logits, spec_decode_metadata

            def _bookkeeping_sync(self, *args: Any, **kwargs: Any) -> Any:
                generator = self.input_batch.generators[1]
                generator.set_offset(generator.get_offset() - 4)
                if self.bookkeeping_failure:
                    raise RuntimeError("bookkeeping failed")
                return args, kwargs

        patch_partial_prefill_rng(Runner)
        patch_partial_prefill_rng(Runner)
        return Runner()

    def test_original_offsets_replace_the_upstream_fixed_four_rewind(self) -> None:
        """The discarded row consumes zero random positions; valid sampling remains intact."""
        runner = self._runner()
        self.assertEqual(runner._sample("logits", None), ("logits", None))
        self.assertEqual(runner._bookkeeping_sync("scheduler", metadata="value"),
                         (("scheduler",), {"metadata": "value"}))
        self.assertEqual(runner.input_batch.generators[0].get_offset(), 44)
        self.assertEqual(runner.input_batch.generators[1].get_offset(), 64)
        self.assertFalse(hasattr(runner, "_hyper_rl_pre_sample_generator_offsets"))

    def test_replaced_generator_rejects_bookkeeping_and_releases_capture(self) -> None:
        """A changed request identity cannot borrow another request's captured RNG state."""
        runner = self._runner()
        runner._sample("logits", None)
        runner.input_batch.generators[1] = OffsetGenerator(100)
        with self.assertRaisesRegex(RuntimeError, "changed a seeded generator"):
            runner._bookkeeping_sync()
        self.assertFalse(hasattr(runner, "_hyper_rl_pre_sample_generator_offsets"))
        self.assertEqual(runner.input_batch.generators[1].get_offset(), 100)

    def test_sample_and_bookkeeping_failures_release_capture(self) -> None:
        """Neither failing stage leaves a stale overlap marker for the next call."""
        for stage in ("sample_failure", "bookkeeping_failure"):
            with self.subTest(stage=stage):
                runner = self._runner()
                setattr(runner, stage, True)
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    runner._sample("logits", None)
                    runner._bookkeeping_sync()
                self.assertFalse(hasattr(runner, "_hyper_rl_pre_sample_generator_offsets"))

    def test_overlapping_sampling_and_unknown_runner_interfaces_fail_closed(self) -> None:
        """Only the reviewed sequential runner hooks can preserve seeded offsets."""
        runner = self._runner()
        runner._sample("logits", None)
        with self.assertRaisesRegex(RuntimeError, "Overlapping"):
            runner._sample("logits", None)
        runner._bookkeeping_sync()
        with self.assertRaisesRegex(ValueError, "_sample"):
            patch_partial_prefill_rng(type("UnknownRunner", (), {}))
