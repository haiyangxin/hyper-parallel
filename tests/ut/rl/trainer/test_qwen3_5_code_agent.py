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
"""Qwen3.5-9B recipes preserve Codex evidence while adopting longer serving budgets."""

import unittest
from copy import deepcopy

from rl.agentic.codex.harness import CodexAgentProgram
from rl.agentic.codex.protocol import CodexResponsesProtocol
from rl.algorithm import build_algorithm
from rl.config import build_model_registration, build_runtime_config, validate_config
from rl.dataset.contracts import Message, PromptRecord
from rl.roles.rollout.vllm import VLLMGenerationEngine

from tests.ut.rl.trainer.code_agent_recipe_fixture import load_code_agent_recipe, prepare_code_agent_fixture


class TestQwen3_5CodeAgent(unittest.TestCase):
    """Exercise both real recipes without weights, services or distributed setup."""

    def setUp(self) -> None:
        """Bind model identity and public task adapters to temporary local data."""
        self.path = prepare_code_agent_fixture(self)

    def _config(self, task: str) -> dict:
        """Read the production recipe and replace its external model/data paths."""
        return load_code_agent_recipe(self.path, "qwen3_5_9b", task)

    def test_recipes_validate_and_build_native_tp1_dp4_serving(self) -> None:
        """Both task types retain public FSDP construction and family-specific native serving."""
        for task in ("code_agent", "swebench"):
            with self.subTest(task=task):
                config = self._config(task)
                validate_config(config, build_algorithm(config["algorithm"]))
                runtime = build_runtime_config(config)
                self.assertEqual(runtime.accelerator.tp_size, 1)
                self.assertEqual(runtime.fsdp_config.dp_shard_size, 4)
                self.assertTrue(runtime.fsdp_config.enable_offload)
                self.assertEqual(runtime.model.to_dict()["_target_"],
                                 "hyper_parallel.models.HyperAutoModelForCausalLM.from_pretrained")
                engine = VLLMGenerationEngine(build_model_registration(config), config["rollout"])
                command = engine._server_command("127.0.0.1", config["rollout"]["vllm"]["port"])
                for flag in ("--language-model-only", "--enforce-eager", "--no-enable-prefix-caching"):
                    self.assertIn(flag, command)
                for flag, expected in (("--tensor-parallel-size", "1"), ("--data-parallel-size", "4"),
                                       ("--tool-call-parser", "qwen3_coder"), ("--logprobs-mode", "raw_logprobs"),
                                       ("--max-model-len", "81920"), ("--max-num-batched-tokens", "8192")):
                    self.assertEqual(command[command.index(flag) + 1], expected)
                self.assertNotIn("--chat-template", command)

    def test_non_thinking_configuration_reaches_existing_codex_protocol(self) -> None:
        """Use the pinned CLI reasoning setting instead of injecting AL's shell agent template."""
        protocol = CodexResponsesProtocol()
        for task in ("code_agent", "swebench"):
            with self.subTest(task=task):
                config = self._config(task)
                codex = config["agentic"]["codex"]
                prompt = PromptRecord("task", (Message("user", "Repair the repository."),), {})
                program = CodexAgentProgram(
                    prompt, 0, 0, "http://127.0.0.1:1", codex, None,
                    admin_url="http://127.0.0.1:1", admin_token="unit-test-controller",
                )
                command = program._codex_command()
                self.assertIn('model_reasoning_effort="none"', command)
                self.assertEqual(command[-1], prompt.messages[-1].content)
                request = protocol.transform_request({
                    "input": command[-1], "reasoning": {"effort": codex["reasoning_effort"]},
                    "max_output_tokens": config["rollout"]["max_new_tokens"],
                }, codex["model"])
                self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": False})
                self.assertEqual(request["max_tokens"], 12288)
                self.assertTrue(request["return_token_ids"])
                self.assertTrue(request["logprobs"])

    def test_long_context_recipes_preserve_evidence_and_publication_guards(self) -> None:
        """Bad context reserve, normalized probabilities or graph publication fail before serving."""
        cases = (
            (("agentic", "codex", "model_context_window"), 81920, "Repository Codex context reserve"),
            (("rollout", "vllm", "logprobs_mode"), "processed_logprobs", "raw_logprobs"),
            (("rollout", "vllm", "tool_call_parser"), "hermes", "tool_call_parser=qwen3_coder"),
            (("rollout", "vllm", "enforce_eager"), False, "enforce_eager"),
        )
        for task in ("code_agent", "swebench"):
            for keys, value, error in cases:
                invalid = deepcopy(self._config(task))
                section = invalid
                for key in keys[:-1]:
                    section = section[key]
                section[keys[-1]] = value
                with self.subTest(task=task, keys=keys), self.assertRaisesRegex(ValueError, error):
                    validate_config(invalid, build_algorithm(invalid["algorithm"]))
