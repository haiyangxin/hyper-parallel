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
"""Request isolation and lifecycle of the optional vLLM evidence hooks."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, AsyncIterator
import unittest
from unittest.mock import patch

from rl import tool_protocol
from tests.common.mark_utils import arg_mark


class TestToolEvidenceHooks(unittest.IsolatedAsyncioTestCase):
    """Exercise installed wrappers without importing vLLM or a model runtime."""

    def setUp(self) -> None:
        """Provide fresh parser/server classes so hook installation cannot leak."""
        self.arrivals = 0
        self.ready = asyncio.Event()
        owner = self

        class Hermes:
            """Small parser result retaining its input for request attribution."""

            def extract_tool_calls(self, model_output: str, request: Any) -> SimpleNamespace:
                """Return an unchanged parser object; validation is tested elsewhere."""
                del request
                return SimpleNamespace(model_dump=lambda: {"tools_called": True, "content": model_output})

        class Qwen:
            """Stand in for the installed reasoning parser's normal split API."""

            def extract_reasoning(self, model_output: str, request: Any) -> tuple[str | None, str]:
                """Produce reasoning only for the explicitly selected fixture mode."""
                if not request.thinking:
                    return None, model_output
                reasoning, content = model_output.split("</think>", 1)
                return reasoning.removeprefix("<think>"), content

        class Server:
            """Suspend after reasoning extraction to force overlapping requests."""

            async def chat_completion_full_generator(
                    self, request: Any, result_generator: AsyncIterator[Any], request_id: str, model_name: str,
                    conversation: Any, tokenizer: Any, request_metadata: Any, reasoning_parser: Any = None,
            ) -> SimpleNamespace:
                """Call the real installed wrappers from an interleaved fake handler."""
                del model_name, conversation, tokenizer, request_metadata
                async for result in result_generator:
                    output = result.outputs[0]
                content = output.text
                if reasoning_parser is not None:
                    parser_request = (SimpleNamespace(thinking=request.thinking)
                                      if request.foreign_reasoning else request)
                    _, content = reasoning_parser.extract_reasoning(content, parser_request)
                if request.interleave:
                    owner.arrivals += 1
                    if owner.arrivals == 2:
                        owner.ready.set()
                    await asyncio.wait_for(owner.ready.wait(), timeout=2)
                Hermes().extract_tool_calls(content, request)
                if request.fail:
                    raise RuntimeError("handler failed after parser capture")
                return SimpleNamespace(model_extra={"retained": request_id})

        paths = {
            "vllm.entrypoints.openai.chat_completion.serving": {"OpenAIServingChat": Server},
            "vllm.tool_parsers.hermes_tool_parser": {"Hermes2ProToolParser": Hermes},
            "vllm.reasoning.qwen3_reasoning_parser": {"Qwen3ReasoningParser": Qwen},
        }
        modules = {}
        for name, attributes in paths.items():
            parts = name.split(".")
            for end in range(1, len(parts) + 1):
                key = ".".join(parts[:end])
                module = modules.setdefault(key, ModuleType(key))
                module.__path__ = []
            modules[name].__dict__.update(attributes)
        mocked = patch.dict(sys.modules, modules)
        mocked.start()
        self.addCleanup(mocked.stop)
        self.server, self.reasoning = Server(), Qwen()
        tool_protocol.install_tool_evidence()
        installed = Server.chat_completion_full_generator
        tool_protocol.install_tool_evidence()
        self.assertIs(Server.chat_completion_full_generator, installed)

    async def _call(self, name: str, *, thinking: bool = True, interleave: bool = False,
                    foreign_reasoning: bool = False, fail: bool = False, capture: bool = True,
                    use_reasoning_parser: bool = True) -> SimpleNamespace:
        """Supply immutable engine IDs and deterministic decoding to one handler."""
        request = SimpleNamespace(return_token_ids=capture, thinking=thinking, interleave=interleave,
                                  foreign_reasoning=foreign_reasoning, fail=fail)
        text = f"<think>reason-{name}</think>tool-{name}" if thinking else f"tool-{name}"
        token_ids = [ord(name), ord(name) + 1]

        async def outputs() -> AsyncIterator[SimpleNamespace]:
            """Model-free async stream matching the serving method's output shape."""
            yield SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=token_ids)])

        def decode(ids: list[int], *, skip_special_tokens: bool) -> str:
            """Ensure capture decodes exactly this request's original sampled IDs."""
            self.assertEqual(ids, token_ids)
            self.assertFalse(skip_special_tokens)
            return text

        return await self.server.chat_completion_full_generator(
            request, outputs(), name, "model", [], SimpleNamespace(decode=decode), None,
            self.reasoning if use_reasoning_parser else None,
        )

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    async def test_concurrent_requests_keep_their_own_reasoning_and_tokens(self) -> None:
        """Interleaving reasoning and Hermes cannot mix request-local evidence."""
        previous = tool_protocol._CAPTURE.get()
        results = await asyncio.gather(self._call("a", interleave=True), self._call("b", interleave=True))
        for name, response in zip(("a", "b"), results):
            extras = response.__pydantic_extra__
            self.assertEqual(extras["retained"], name)
            self.assertEqual(len(extras["hyper_tool_protocol"]), 1)
            evidence = extras["hyper_tool_protocol"][0]
            self.assertEqual(evidence["request_id"], name)
            self.assertEqual(evidence["token_ids"], [ord(name), ord(name) + 1])
            self.assertEqual(evidence["parser_input"], f"tool-{name}")
            self.assertEqual(evidence["reasoning_parser"]["reasoning"], f"reason-{name}")
            self.assertEqual(evidence["reasoning_parser"]["content"], f"tool-{name}")
            self.assertEqual(evidence["decoded_tokens"], evidence["engine_text"])
        self.assertIs(tool_protocol._CAPTURE.get(), previous)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    async def test_foreign_request_reasoning_is_marked_invalid(self) -> None:
        """A parser result for another request cannot authenticate this response."""
        response = await self._call("a", foreign_reasoning=True)
        evidence = response.__pydantic_extra__["hyper_tool_protocol"][0]
        self.assertEqual(evidence["reasoning_parser"], {"invalid": "cross_request_reasoning_capture"})

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    async def test_failure_restores_context_and_legacy_capture_is_unchanged(self) -> None:
        """Exceptions and non-instrumented requests leave no stale reasoning evidence."""
        previous = tool_protocol._CAPTURE.get()
        with self.assertRaisesRegex(RuntimeError, "handler failed"):
            await self._call("a", fail=True)
        self.assertIs(tool_protocol._CAPTURE.get(), previous)
        legacy = await self._call("b", thinking=False)
        evidence = legacy.__pydantic_extra__["hyper_tool_protocol"][0]
        self.assertEqual(evidence["parser_input"], "tool-b")
        self.assertNotIn("reasoning_parser", evidence)
        no_parser = await self._call("b", thinking=False, use_reasoning_parser=False)
        self.assertNotIn("reasoning_parser", no_parser.__pydantic_extra__["hyper_tool_protocol"][0])
        disabled = await self._call("c", capture=False)
        self.assertFalse(hasattr(disabled, "__pydantic_extra__"))
        self.assertEqual(disabled.model_extra, {"retained": "c"})
        self.assertIs(tool_protocol._CAPTURE.get(), previous)
