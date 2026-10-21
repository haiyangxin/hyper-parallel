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
"""GSM8K comparison environment retaining numeric and verl-format scores."""

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from rl.agentic.core.types import Action, EpisodeContext, Transition, TurnContext
from rl.agentic.envs.environment import ENVIRONMENTS
from rl.dataset.contracts import PromptRecord

from examples.gsm8k.agent import GSM8KSingleTurnEnvironment, compute_gsm8k_reward


def compute_verl_reward(answer: str, ground_truth: str) -> float:
    """Match verl's strict last-300-character GSM8K scoring."""
    solutions = re.findall(r"#### (\-?[0-9\.\,]+)", answer[-300:])
    if not solutions:
        return 0.0
    prediction = solutions[-1].replace(",", "").replace("$", "")
    return float(prediction == ground_truth)


class ComparisonGSM8KEnvironment(GSM8KSingleTurnEnvironment):
    """Train with the existing numeric scorer and retain both scoring conventions."""

    @staticmethod
    def score_reward(
        prompt: PromptRecord, *, answer: Optional[str] = None, **kwargs: Any,
    ) -> float:
        """Score mathematically equivalent numeric answers consistently."""
        if answer is None or kwargs.get("reward_model") is not None:
            raise ValueError("The verl comparison requires a rule-scored answer")
        return compute_gsm8k_reward(answer, str(prompt.ground_truth))

    async def step(self, action: Action, context: TurnContext) -> Transition:
        """Retain answers and both scoring conventions for comparison audits."""
        transition = await super().step(action, context)
        output_dir = os.environ.get("HYPER_COMPARISON_OUTPUT")
        if output_dir:
            directory = Path(output_dir) / "rollouts"
            directory.mkdir(parents=True, exist_ok=True)
            record = {
                "prompt_id": self.prompt.prompt_id,
                "input": self.prompt.messages[-1].content,
                "output": action.content,
                "gts": str(self.prompt.ground_truth),
                "score": transition.reward,
                "verl_score": compute_verl_reward(action.content, str(self.prompt.ground_truth)),
                "phase": self.prompt.metadata.get("phase", "training"),
                "sample_index": self.episode.sample_index,
                "policy_version": self.episode.policy_version,
                "response_tokens": int(action.token_ids.numel()),
            }
            with (directory / f"rank-{os.environ.get('RANK', '0')}.jsonl").open(
                "a", encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return transition


@ENVIRONMENTS.register("gsm8k_comparison")
def build_verl_environment(context: EpisodeContext) -> ComparisonGSM8KEnvironment:
    """Construct the comparison-only environment."""
    return ComparisonGSM8KEnvironment(context)


class StrictComparisonGSM8KEnvironment(ComparisonGSM8KEnvironment):
    """Use verl's strict training reward while retaining the same comparison records."""

    @staticmethod
    def score_reward(
        prompt: PromptRecord, *, answer: Optional[str] = None, **kwargs: Any,
    ) -> float:
        """Apply the same strict final-answer rule to training and evaluation."""
        if answer is None or kwargs.get("reward_model") is not None:
            raise ValueError("The strict comparison requires a rule-scored answer")
        return compute_verl_reward(answer, str(prompt.ground_truth))


@ENVIRONMENTS.register("gsm8k_comparison_strict")
def build_strict_verl_environment(context: EpisodeContext) -> StrictComparisonGSM8KEnvironment:
    """Construct the opt-in strict-reward comparison environment."""
    return StrictComparisonGSM8KEnvironment(context)
