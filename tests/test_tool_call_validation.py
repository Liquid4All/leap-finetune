from leap_finetune.data_loaders.tool_call_utils import (
    ToolFormatInfo,
    detect_tool_format,
    normalize_tool_format,
    validate_tool_format,
    _tool_calls_to_pythonic,
)
from leap_finetune.utils.model_utils import get_model_family


# === Model family detection ===


class TestGetModelFamily:
    def test_lfm2_models(self):
        assert get_model_family("LFM2-1.2B") == "lfm2"
        assert get_model_family("LFM2-2.6B") == "lfm2"
        assert get_model_family("LFM2-8B-A1B") == "lfm2"
        assert get_model_family("LFM2-350M") == "lfm2"

    def test_lfm25_models(self):
        assert get_model_family("LFM2.5-1.2B-Instruct") == "lfm25"
        assert get_model_family("LFM2.5-8B-A1B-Instruct") == "lfm25"

    def test_24b_exception(self):
        assert get_model_family("LFM2-24B-A2B") == "lfm25"

    def test_full_path(self):
        assert get_model_family("LiquidAI/LFM2-1.2B") == "lfm2"
        assert get_model_family("LiquidAI/LFM2.5-1.2B-Instruct") == "lfm25"


# === Format detection ===


class TestDetectToolFormat:
    def test_lfm2_format(self):
        samples = [
            {
                "conversations": [
                    {
                        "role": "system",
                        "content": '<|tool_list_start|>[{"name": "f"}]<|tool_list_end|>',
                    },
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "<|tool_call_start|>[f()]<|tool_call_end|>",
                    },
                    {"role": "tool", "content": '{"result": 1}'},
                ]
            }
        ]
        info = detect_tool_format(samples)
        assert info.detected_format == "lfm2"
        assert info.has_tool_call_markers
        assert info.has_tool_list_markers
        assert info.has_tool_role

    def test_lfm25_format(self):
        samples = [
            {
                "conversations": [
                    {"role": "system", "content": 'List of tools: [{"name": "f"}]'},
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "<|tool_call_start|>[f()]<|tool_call_end|>",
                    },
                    {"role": "tool", "content": '{"result": 1}'},
                ]
            }
        ]
        info = detect_tool_format(samples)
        assert info.detected_format == "lfm25"
        assert info.has_tool_call_markers
        assert not info.has_tool_list_markers

    def test_structured_format(self):
        samples = [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "f", "arguments": {}},
                            }
                        ],
                    },
                ]
            }
        ]
        info = detect_tool_format(samples)
        assert info.detected_format == "structured"
        assert info.has_structured_tool_calls

    def test_foreign_format(self):
        samples = [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "<tool_call>get_weather</tool_call>",
                    },
                ]
            }
        ]
        info = detect_tool_format(samples)
        assert info.detected_format == "foreign"
        assert info.has_foreign_markers
        assert "<tool_call>" in info.foreign_markers_found

    def test_no_tools(self):
        samples = [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                ]
            }
        ]
        info = detect_tool_format(samples)
        assert info.detected_format == "none"
        assert not info.has_tool_calls

    def test_prebaked_tool_response(self):
        samples = [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "<|tool_call_start|>[f()]<|tool_call_end|>",
                    },
                    {
                        "role": "tool",
                        "content": '<|tool_response_start|>{"r": 1}<|tool_response_end|>',
                    },
                ]
            }
        ]
        info = detect_tool_format(samples)
        assert info.has_tool_response_markers


# === Validation ===


class TestValidateToolFormat:
    def test_matched_lfm2(self):
        info = ToolFormatInfo(
            has_tool_calls=True,
            has_tool_call_markers=True,
            has_tool_list_markers=True,
            detected_format="lfm2",
        )
        issues = validate_tool_format(info, "lfm2")
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_matched_lfm25(self):
        info = ToolFormatInfo(
            has_tool_calls=True,
            has_tool_call_markers=True,
            detected_format="lfm25",
        )
        issues = validate_tool_format(info, "lfm25")
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_lfm2_on_lfm25_warns(self):
        info = ToolFormatInfo(
            has_tool_calls=True,
            has_tool_call_markers=True,
            has_tool_list_markers=True,
            detected_format="lfm2",
        )
        issues = validate_tool_format(info, "lfm25")
        warnings = [i for i in issues if i.severity == "warning"]
        assert any(i.code == "TOOL_LIST_MARKERS_ON_LFM25" for i in warnings)
        assert all(i.auto_fixable for i in warnings)

    def test_lfm25_on_lfm2_warns(self):
        info = ToolFormatInfo(
            has_tool_calls=True,
            has_tool_call_markers=True,
            detected_format="lfm25",
        )
        issues = validate_tool_format(info, "lfm2")
        warnings = [i for i in issues if i.severity == "warning"]
        assert any(i.code == "MISSING_TOOL_LIST_MARKERS" for i in warnings)

    def test_structured_auto_fixable(self):
        info = ToolFormatInfo(
            has_tool_calls=True,
            has_structured_tool_calls=True,
            detected_format="structured",
        )
        issues = validate_tool_format(info, "lfm2")
        assert any(i.code == "STRUCTURED_TOOL_CALLS" and i.auto_fixable for i in issues)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_foreign_errors(self):
        info = ToolFormatInfo(
            has_tool_calls=True,
            has_foreign_markers=True,
            foreign_markers_found=["<tool_call>", "</tool_call>"],
            detected_format="foreign",
        )
        issues = validate_tool_format(info, "lfm2")
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 1
        assert errors[0].code == "FOREIGN_MARKERS"
        assert not errors[0].auto_fixable

    def test_prebaked_response_info(self):
        info = ToolFormatInfo(
            has_tool_calls=True,
            has_tool_call_markers=True,
            has_tool_response_markers=True,
            detected_format="lfm25",
        )
        issues = validate_tool_format(info, "lfm25")
        assert any(i.code == "PREBAKED_TOOL_RESPONSE" for i in issues)


# === Pythonic conversion ===


class TestToolCallsToPythonic:
    def test_simple(self):
        tc = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "Boston"},
                },
            }
        ]
        result = _tool_calls_to_pythonic(tc)
        assert (
            result
            == '<|tool_call_start|>[get_weather(location="Boston")]<|tool_call_end|>'
        )

    def test_multiple_args(self):
        tc = [
            {
                "type": "function",
                "function": {"name": "f", "arguments": {"a": "x", "b": 42}},
            }
        ]
        result = _tool_calls_to_pythonic(tc)
        assert result == '<|tool_call_start|>[f(a="x", b=42)]<|tool_call_end|>'

    def test_multiple_calls(self):
        tc = [
            {"type": "function", "function": {"name": "f1", "arguments": {"a": 1}}},
            {"type": "function", "function": {"name": "f2", "arguments": {"b": "y"}}},
        ]
        result = _tool_calls_to_pythonic(tc)
        assert result == '<|tool_call_start|>[f1(a=1), f2(b="y")]<|tool_call_end|>'

    def test_no_args(self):
        tc = [{"type": "function", "function": {"name": "get_time", "arguments": {}}}]
        result = _tool_calls_to_pythonic(tc)
        assert result == "<|tool_call_start|>[get_time()]<|tool_call_end|>"

    def test_string_arguments(self):
        tc = [
            {
                "type": "function",
                "function": {"name": "f", "arguments": '{"x": "hello"}'},
            }
        ]
        result = _tool_calls_to_pythonic(tc)
        assert result == '<|tool_call_start|>[f(x="hello")]<|tool_call_end|>'

    def test_bool_arg(self):
        tc = [
            {"type": "function", "function": {"name": "f", "arguments": {"flag": True}}}
        ]
        result = _tool_calls_to_pythonic(tc)
        assert result == "<|tool_call_start|>[f(flag=True)]<|tool_call_end|>"


# === Normalization ===


class TestNormalizeToolFormat:
    def test_strip_tool_list_for_lfm25(self):
        row = {
            "conversations": [
                {
                    "role": "system",
                    "content": '<|tool_list_start|>[{"name": "f"}]<|tool_list_end|>\nYou help.',
                },
                {"role": "user", "content": "hi"},
            ]
        }
        result = normalize_tool_format(row, "lfm25")
        system_content = result["conversations"][0]["content"]
        assert "<|tool_list_start|>" not in system_content
        assert "<|tool_list_end|>" not in system_content
        assert '{"name": "f"}' in system_content

    def test_keep_tool_list_for_lfm2(self):
        row = {
            "conversations": [
                {
                    "role": "system",
                    "content": '<|tool_list_start|>[{"name": "f"}]<|tool_list_end|>',
                },
                {"role": "user", "content": "hi"},
            ]
        }
        result = normalize_tool_format(row, "lfm2")
        system_content = result["conversations"][0]["content"]
        assert "<|tool_list_start|>" in system_content

    def test_strip_tool_response_markers(self):
        row = {
            "messages": [
                {
                    "role": "tool",
                    "content": '<|tool_response_start|>{"r": 1}<|tool_response_end|>',
                },
            ]
        }
        result = normalize_tool_format(row, "lfm2")
        assert result["messages"][0]["content"] == '{"r": 1}'

    def test_strip_tool_response_for_lfm25_too(self):
        row = {
            "messages": [
                {
                    "role": "tool",
                    "content": '<|tool_response_start|>{"r": 1}<|tool_response_end|>',
                },
            ]
        }
        result = normalize_tool_format(row, "lfm25")
        assert result["messages"][0]["content"] == '{"r": 1}'

    def test_convert_structured_tool_calls(self):
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"location": "SF"},
                            },
                        },
                    ],
                },
            ]
        }
        result = normalize_tool_format(row, "lfm2")
        assistant_msg = result["messages"][1]
        assert "<|tool_call_start|>" in assistant_msg["content"]
        assert 'get_weather(location="SF")' in assistant_msg["content"]
        assert "tool_calls" not in assistant_msg

    def test_no_modification_when_clean(self):
        row = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        }
        result = normalize_tool_format(row, "lfm2")
        assert result is row  # Same object, no copy

    def test_preserves_column_name(self):
        row_conv = {
            "conversations": [
                {"role": "system", "content": "<|tool_list_start|>x<|tool_list_end|>"}
            ]
        }
        result = normalize_tool_format(row_conv, "lfm25")
        assert "conversations" in result
        assert "messages" not in result

    def test_no_double_wrapping_after_normalize_and_template(self):
        """After normalization, LFM2 template should produce exactly one tool_response wrapper."""
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "<|tool_call_start|>[f()]<|tool_call_end|>",
                },
                {
                    "role": "tool",
                    "content": '<|tool_response_start|>{"r": 1}<|tool_response_end|>',
                },
            ]
        }
        result = normalize_tool_format(row, "lfm2")
        # After normalization, tool content should be clean
        assert result["messages"][2]["content"] == '{"r": 1}'
