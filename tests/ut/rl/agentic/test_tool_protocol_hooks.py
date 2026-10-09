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
"""Real wrapper semantics for vLLM's unified full-response parser interface."""
# White-box context checks ensure failed requests cannot leak evidence into another request.
# pylint: disable=protected-access

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, AsyncIterator
import unittest
from unittest.mock import patch

from rl import tool_protocol


class TestUnifiedParserEvidence(unittest.IsolatedAsyncioTestCase):
    """Capture genuine parser invocations without importing a vLLM runtime."""

    def setUp(self) -> None:
        """Provide the reviewed 0.23 public Parser property and generator signature."""
        class Hermes:
            """Preserve real input text in the fixture parser result."""

            def extract_tool_calls(self, model_output: str, request: Any) -> SimpleNamespace:
                """Return a parser object without replacing its input."""
                del request
                return SimpleNamespace(model_dump=lambda: {"tools_called": True, "content": model_output})

        class Qwen:
            """Stand in for the normal reasoning extraction delegate."""

            def extract_reasoning(self, model_output: str, request: Any) -> tuple[str, str]:
                """Split actual text only when the fixture emits reasoning."""
                del request
                reason, content = model_output.split("</think>", 1)
                return reason.removeprefix("<think>"), content

        class Coder:
            """Expose the additional reviewed 0.23 delegate without native dependencies."""

            def extract_tool_calls(self, model_output: str, request: Any) -> SimpleNamespace:
                """Preserve the actual XML fixture delegate input."""
                if getattr(request, "mutate_schema", False):
                    request.tools[0]["function"]["parameters"]["properties"]["a"]["type"] = "string"
                return SimpleNamespace(model_dump=lambda: {"tools_called": False, "content": model_output})

        class Server:
            """Mirror the upstream full-generator parser argument and delegation."""

            async def chat_completion_full_generator(
                self, request: Any, result_generator: AsyncIterator[Any], request_id: str, model_name: str,
                conversation: Any, tokenizer: Any, request_metadata: Any, parser: Any = None,
            ) -> SimpleNamespace:
                """Observe output and invoke exactly the selected parser delegates."""
                del model_name, conversation, tokenizer, request_metadata
                async for result in result_generator:
                    content = result.outputs[0].text
                if parser.reasoning_parser is not None:
                    _, content = parser.reasoning_parser.extract_reasoning(content, request)
                getattr(parser, "tool_parser", Hermes()).extract_tool_calls(content, request)
                await asyncio.sleep(0)
                if request.fail:
                    raise RuntimeError("handler failed")
                return SimpleNamespace(model_extra={"request_id": request_id})

        definitions = {
            "vllm.entrypoints.openai.chat_completion.serving": {"OpenAIServingChat": Server},
            "vllm.tool_parsers.hermes_tool_parser": {"Hermes2ProToolParser": Hermes},
            "vllm.reasoning.qwen3_reasoning_parser": {"Qwen3ReasoningParser": Qwen},
            "vllm.tool_parsers.qwen3coder_tool_parser": {"Qwen3CoderToolParser": Coder},
        }
        modules = {}
        for name, attributes in definitions.items():
            parts = name.split(".")
            for length in range(1, len(parts) + 1):
                module = modules.setdefault(".".join(parts[:length]), ModuleType(".".join(parts[:length])))
                module.__path__ = []
            modules[name].__dict__.update(attributes)
        mocked = patch.dict(sys.modules, modules)
        mocked.start()
        self.addCleanup(mocked.stop)
        self.server, self.reasoning = Server(), Qwen()
        tool_protocol.install_tool_evidence()

    async def _call(self, label: str, *, reasoning: bool = False, keyword: bool = True,
                    fail: bool = False) -> SimpleNamespace:
        """Pass one request and unmodified engine tokens through the installed wrapper."""
        request = SimpleNamespace(return_token_ids=True, fail=fail)
        text = f"<think>reason-{label}</think>tool-{label}" if reasoning else f"tool-{label}"
        ids = [ord(label)]

        async def outputs() -> AsyncIterator[SimpleNamespace]:
            """Emit one immutable fixture engine output."""
            yield SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=ids)])

        tokenizer = SimpleNamespace(decode=lambda token_ids, **kwargs: text)
        parser = SimpleNamespace(reasoning_parser=self.reasoning if reasoning else None)
        arguments = (request, outputs(), label, "model", [], tokenizer, None)
        if keyword:
            return await self.server.chat_completion_full_generator(*arguments, parser=parser)
        return await self.server.chat_completion_full_generator(*arguments, parser)

    async def test_tool_only_parser_is_not_mistaken_for_reasoning_parser(self) -> None:
        """A unified tool-only parser must not create invalid missing-reasoning evidence."""
        response = await self._call("a")
        evidence = response.__pydantic_extra__["hyper_tool_protocol"][0]
        self.assertNotIn("reasoning_parser", evidence)
        self.assertEqual(evidence["engine_text"], "tool-a")
        self.assertEqual(evidence["token_ids"], [ord("a")])

    async def test_keyword_and_positional_unified_parser_capture_actual_reasoning(self) -> None:
        """Both upstream positional calls and public keyword calls retain actual delegates."""
        responses = await asyncio.gather(self._call("a", reasoning=True),
                                         self._call("b", reasoning=True, keyword=False))
        for name, response in zip(("a", "b"), responses):
            evidence = response.__pydantic_extra__["hyper_tool_protocol"][0]
            self.assertEqual(evidence["request_id"], name)
            self.assertEqual(evidence["parser_input"], f"tool-{name}")
            self.assertEqual(evidence["reasoning_parser"]["reasoning"], f"reason-{name}")
            self.assertEqual(evidence["token_ids"], [ord(name)])

    async def test_handler_failure_does_not_leak_request_capture(self) -> None:
        """A failed handler always releases its context before the next completion."""
        previous = tool_protocol._CAPTURE.get()
        with self.assertRaisesRegex(RuntimeError, "handler failed"):
            await self._call("a", reasoning=True, fail=True)
        self.assertIs(tool_protocol._CAPTURE.get(), previous)
        response = await self._call("b")
        self.assertEqual(response.__pydantic_extra__["hyper_tool_protocol"][0]["request_id"], "b")

    def test_unknown_generator_interface_leaves_all_delegates_unchanged(self) -> None:
        """Reject an unsupported generator before installing partial instrumentation."""
        server_type = type(self.server)
        hermes = sys.modules["vllm.tool_parsers.hermes_tool_parser"].Hermes2ProToolParser
        reasoner = type(self.reasoning)
        original_parse = hermes.extract_tool_calls
        original_reasoning = reasoner.extract_reasoning

        async def unsupported(*args: Any, **kwargs: Any) -> None:
            """Provide an unreviewed generator signature that cannot expose its parser."""
            del args, kwargs

        server_type.chat_completion_full_generator = unsupported
        with self.assertRaisesRegex(ValueError, "no supported parser parameter"):
            tool_protocol.install_tool_evidence()
        self.assertIs(server_type.chat_completion_full_generator, unsupported)
        self.assertIs(hermes.extract_tool_calls, original_parse)
        self.assertIs(reasoner.extract_reasoning, original_reasoning)

    async def test_xml_delegate_keeps_request_schema_before_parser_mutation(self) -> None:
        """The captured schema comes from the request before the real delegate executes."""
        coder = sys.modules["vllm.tool_parsers.qwen3coder_tool_parser"].Qwen3CoderToolParser
        request = SimpleNamespace(return_token_ids=True, fail=False, mutate_schema=True, tools=[{
            "type": "function", "function": {"name": "add_numbers", "parameters": {
                "type": "object", "properties": {"a": {"type": "integer"}}}}}])
        raw = "<tool_call><function=add_numbers><parameter=a>1</parameter></function></tool_call>"

        async def outputs() -> AsyncIterator[SimpleNamespace]:
            """Provide unchanged engine IDs and text to the fake serving delegate."""
            yield SimpleNamespace(outputs=[SimpleNamespace(text=raw, token_ids=[7, 8])])

        parser = SimpleNamespace(reasoning_parser=None, tool_parser=coder())
        response = await self.server.chat_completion_full_generator(
            request, outputs(), "xml", "model", [], SimpleNamespace(decode=lambda *args, **kwargs: raw), None,
            parser=parser)
        evidence = response.__pydantic_extra__["hyper_tool_protocol"][0]
        self.assertEqual(evidence["source"], "Qwen3CoderToolParser.extract_tool_calls")
        self.assertEqual(evidence["parser_format"], "qwen3_xml")
        self.assertEqual(evidence["token_ids"], [7, 8])
        self.assertEqual(evidence["engine_text"], raw)
        self.assertEqual(evidence["request_tools"][0]["function"]["parameters"]["properties"]["a"]["type"], "integer")
        self.assertEqual(request.tools[0]["function"]["parameters"]["properties"]["a"]["type"], "string")
