import json

import pytest

from trl.data_utils import maybe_apply_chat_template

pytestmark = pytest.mark.data

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]


class TestLFMToolCallTemplate:
    """Empirically document what apply_chat_template auto-formats for LFM tokenizers.

    Key findings from reading the LFM2-1.2B chat template:

    1. The template ONLY reads message["content"] — it completely ignores
       message["tool_calls"]. Structured tool_calls are NOT supported.
    2. tools= parameter → <|tool_list_start|>[...]<|tool_list_end|> in system msg
    3. role="tool" → wraps content with <|tool_response_start|>/<|tool_response_end|>
    4. Content is passed through VERBATIM — no reordering, no marker injection

    This means tool call data MUST be pre-formatted in the content field:
    - Tool calls must use <|tool_call_start|>...<|tool_call_end|> in content
    - Tool calls must appear BEFORE any text in content
    - Foreign markers (Qwen <tool_call>, Mistral [TOOL_CALLS]) will pass through
      as plain text and NOT be converted to LFM format
    """

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from leap_finetune.utils.load_models import load_tokenizer

        return load_tokenizer("LFM2-1.2B")

    # === 1. Template ignores structured tool_calls ===

    def test_structured_tool_calls_crashes_without_content(self, tokenizer):
        """Structured tool_calls without content field crashes the template.
        The template tries to access message["content"] which is Undefined.
        """
        messages = [
            {"role": "user", "content": "What's the weather in SF?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "SF"}),
                        },
                    }
                ],
            },
        ]
        with pytest.raises(TypeError, match="Undefined"):
            tokenizer.apply_chat_template(messages, tools=SAMPLE_TOOLS, tokenize=False)

    def test_structured_tool_calls_ignored_when_content_present(self, tokenizer):
        """When both content and tool_calls are present, tool_calls is ignored.
        Only the content text appears in the output.
        """
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "Let me check that for you.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "SF"}),
                        },
                    }
                ],
            },
        ]
        output = tokenizer.apply_chat_template(
            messages, tools=SAMPLE_TOOLS, tokenize=False
        )
        # tool_calls field is completely ignored — no tool call markers generated
        assert "<|tool_call_start|>" not in output
        # Only content text appears in the assistant turn
        assert "Let me check that for you." in output

    # === 2. Pre-formatted LFM markers in content ===

    def test_preformatted_lfm_markers_pass_through(self, tokenizer):
        """Content with LFM tool call markers passes through verbatim."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": '<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>',
            },
        ]
        output = tokenizer.apply_chat_template(messages, tokenize=False)
        count = output.count("<|tool_call_start|>")
        assert count == 1, (
            f"Expected 1 occurrence of <|tool_call_start|>, found {count} in:\n{output}"
        )

    def test_preformatted_tool_call_then_text(self, tokenizer):
        """Correct ordering: tool call first, then text. Passes through as-is."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": '<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>\nChecking now.',
            },
        ]
        output = tokenizer.apply_chat_template(messages, tokenize=False)
        tool_pos = output.find("<|tool_call_start|>")
        text_pos = output.find("Checking now.")
        assert tool_pos < text_pos

    def test_preformatted_wrong_order_passes_through(self, tokenizer):
        """Wrong ordering (text first) also passes through — template does NOT fix it.
        This is the bug that validation needs to catch.
        """
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": 'Checking now.\n<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>',
            },
        ]
        output = tokenizer.apply_chat_template(messages, tokenize=False)
        tool_pos = output.find("<|tool_call_start|>")
        text_pos = output.find("Checking now.")
        assert text_pos < tool_pos, (
            "Expected wrong order to pass through, but template reordered it"
        )

    # === 3. Foreign markers ===

    def test_qwen_markers_pass_through_unchanged(self, tokenizer):
        """Qwen-style <tool_call> markers pass through as plain text.
        The template does NOT convert them to LFM format.
        """
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": '<tool_call>\n{"name": "get_weather"}\n</tool_call>',
            },
        ]
        output = tokenizer.apply_chat_template(messages, tokenize=False)
        assert "<tool_call>" in output
        assert "</tool_call>" in output
        assert "<|tool_call_start|>" not in output

    # === 4. tools= parameter ===

    def test_tools_parameter_produces_tool_list_markers(self, tokenizer):
        """tools= kwarg adds tool definitions to system message."""
        messages = [{"role": "user", "content": "What's the weather?"}]
        output = tokenizer.apply_chat_template(
            messages, tools=SAMPLE_TOOLS, tokenize=False
        )
        assert "<|tool_list_start|>" in output
        assert "<|tool_list_end|>" in output
        assert "get_weather" in output

    # === 5. role="tool" ===

    def test_tool_role_produces_response_markers(self, tokenizer):
        """role='tool' wraps content with tool response special tokens."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": '<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>',
            },
            {"role": "tool", "content": '{"temp": 18, "unit": "celsius"}'},
        ]
        output = tokenizer.apply_chat_template(messages, tokenize=False)
        assert "<|tool_response_start|>" in output
        assert "<|tool_response_end|>" in output

    def test_tool_role_without_tool_call_in_prior_message(self, tokenizer):
        """role='tool' works even without a prior tool call (template doesn't validate)."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "tool", "content": '{"result": "ok"}'},
        ]
        output = tokenizer.apply_chat_template(messages, tokenize=False)
        assert "<|tool_response_start|>" in output
        assert "<|tool_response_end|>" in output

    # === 6. DPO path via TRL ===

    def test_trl_string_data_passes_through(self, tokenizer):
        """TRL with pre-stringified DPO data passes strings through unchanged."""
        row = {
            "prompt": "What's the weather?",
            "chosen": '<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>',
            "rejected": "I don't know the weather.",
        }
        result = maybe_apply_chat_template(row, tokenizer)
        assert result["chosen"] == row["chosen"]
        assert result["rejected"] == row["rejected"]

    def test_trl_structured_tool_calls_not_converted(self, tokenizer):
        """TRL fails to convert structured tool_calls — either crashes or
        silently returns data unchanged (behavior depends on TRL internals).
        Either way, the data is NOT properly converted to strings with LFM markers.
        This confirms DPO tool call data must be pre-formatted in content.
        """
        row = {
            "prompt": [{"role": "user", "content": "What's the weather?"}],
            "chosen": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "SF"}),
                            },
                        }
                    ],
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": "I don't have access to weather data.",
                }
            ],
        }
        try:
            result = maybe_apply_chat_template(row, tokenizer)
        except (TypeError, Exception):
            # Template crash — structured tool_calls unsupported
            return

        # If it didn't crash, data should be unchanged (not converted to strings)
        # Either way, <|tool_call_start|> won't appear in the chosen output
        if isinstance(result["chosen"], str):
            assert "<|tool_call_start|>" not in result["chosen"], (
                "Structured tool_calls should not produce LFM markers via TRL"
            )
