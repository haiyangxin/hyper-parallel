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
"""Native Qwen3.8 repository recipes reuse the public text model and captured-call budgets."""

import unittest
from copy import deepcopy

from rl.algorithm import build_algorithm
from rl.config import build_model_registration, build_runtime_config, validate_config
from rl.roles.rollout.vllm import VLLMGenerationEngine

from tests.ut.rl.trainer.code_agent_recipe_fixture import load_code_agent_recipe, prepare_code_agent_fixture


class TestQwen3_8CodeAgent(unittest.TestCase):
    """Validate supported repository integration without loading weights or starting a service."""

    def setUp(self) -> None:
        """Bind recipe startup validation to local checkpoint and public data fixtures."""
        self.path = prepare_code_agent_fixture(self)

    def _config(self, task: str) -> dict:
        """Read the actual new repository recipe, replacing only external model paths."""
        return load_code_agent_recipe(self.path, "qwen3_8_27b", task)

    def test_repository_recipes_build_public_text_training_and_native_language_only_serving(self) -> None:
        """Both task adapters pass production validation and produce the supported native server command."""
        for task in ("code_agent", "swebench"):
            with self.subTest(task=task):
                config = self._config(task)
                validate_config(config, build_algorithm(config["algorithm"]))
                runtime = build_runtime_config(config)
                target = runtime.model.to_dict()
                self.assertEqual(target["_target_"], "hyper_parallel.models.HyperAutoModelForCausalLM.from_pretrained")
                self.assertNotIn("fused", target)
                self.assertEqual(runtime.accelerator.tp_size, 1)
                self.assertEqual(runtime.fsdp_config.dp_shard_size, 4)
                self.assertTrue(runtime.fsdp_config.enable_offload)
                engine = VLLMGenerationEngine(build_model_registration(config), config["rollout"])
                command = engine._server_command("127.0.0.1", config["rollout"]["vllm"]["port"])
                self.assertIn("--language-model-only", command)
                self.assertIn("--enforce-eager", command)
                self.assertIn("--enable-auto-tool-choice", command)
                self.assertEqual(command[command.index("--tool-call-parser") + 1], "qwen3_coder")
                self.assertNotIn("--hf-overrides", command)
                self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")
                self.assertEqual(command[command.index("--data-parallel-size") + 1], "2")
                self.assertEqual(command[command.index("--served-model-name") + 1], config["agentic"]["codex"]["model"])

    def test_repository_context_reserve_and_raw_probability_guards_remain_active(self) -> None:
        """Invalid output reserve or normalized probabilities reject the new recipes before inference."""
        for task in ("code_agent", "swebench"):
            config = self._config(task)
            for mutation in ("context", "logprobs"):
                invalid = deepcopy(config)
                if mutation == "context":
                    invalid["agentic"]["codex"]["model_context_window"] = invalid["rollout"]["vllm"]["max_model_len"]
                    error = "Repository Codex context reserve"
                else:
                    invalid["rollout"]["vllm"]["logprobs_mode"] = "processed_logprobs"
                    error = "raw_logprobs"
                with self.subTest(task=task, mutation=mutation), self.assertRaisesRegex(ValueError, error):
                    validate_config(invalid, build_algorithm(invalid["algorithm"]))
