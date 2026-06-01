import ast
from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

from leap_finetune.data_loading.validate_tool_format import (
    normalize_messages_for_chat_template,
)


LFM2_5_TEMPLATE_PATH = Path("job_configs/chat_templates/lfm2_5_chat_template.jinja")
LEGACY_LFM2_TEMPLATE_PATH = Path("job_configs/chat_templates/lfm2_chat_template.jinja")

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search documents",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def _render(messages, template_path=LFM2_5_TEMPLATE_PATH, tools=None):
    rendered, _ = render_jinja_template(
        [messages],
        chat_template=template_path.read_text(),
        tools=tools,
        add_generation_prompt=False,
        bos_token="<s>",
        strftime_now=lambda _: "2026-05-18",
    )
    return rendered[0]


def _tool_call_payload(rendered: str) -> str:
    start = rendered.index("<|tool_call_start|>") + len("<|tool_call_start|>")
    end = rendered.index("<|tool_call_end|>")
    return rendered[start:end]


def _render_with_assistant_indices(messages, template_path):
    rendered, generation_indices = render_jinja_template(
        [messages],
        chat_template=template_path.read_text(),
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
        bos_token="<s>",
        strftime_now=lambda _: "2026-05-18",
    )
    return rendered[0], generation_indices[0]


def test_string_tool_arguments_are_escaped_as_valid_python_literals():
    rendered = _render(
        [
            {"role": "user", "content": "find shoes"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": {
                                "query": "women's shoes",
                                "notes": 'quote " and newline\nok',
                            },
                        },
                    }
                ],
            },
        ]
    )

    payload = _tool_call_payload(rendered)
    ast.parse(payload, mode="eval")
    assert "search(query=" in payload
    assert "query='women's shoes'" not in payload


def test_null_assistant_content_with_tool_calls_renders_empty_content():
    rendered = _render(
        [
            {"role": "user", "content": "find shoes"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "search", "arguments": {"query": "shoes"}}}
                ],
            },
        ]
    )

    assert "<|tool_call_start|>[search(query=" in rendered
    assert "<|tool_call_end|>" in rendered


def test_flat_tool_call_schema_renders_like_openai_nested_schema():
    rendered = _render(
        [
            {"role": "user", "content": "find shoes"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "search", "arguments": {"query": "shoes"}}],
            },
        ]
    )

    assert '<|tool_call_start|>[search(query="shoes")]<|tool_call_end|>' in rendered


def test_json_string_tool_arguments_are_normalized_before_template_rendering():
    messages = normalize_messages_for_chat_template(
        [
            {"role": "user", "content": "find shoes"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "women\'s shoes"}',
                        }
                    }
                ],
            },
        ]
    )

    rendered = _render(messages)
    payload = _tool_call_payload(rendered)

    ast.parse(payload, mode="eval")
    assert 'search(query="women\'s shoes")' in payload


def test_template_requires_normalized_mapping_arguments():
    with pytest.raises(Exception):
        _render(
            [
                {"role": "user", "content": "find shoes"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "shoes"}',
                            }
                        }
                    ],
                },
            ]
        )


def test_lfm2_5_renders_assistant_content_before_structured_tool_calls():
    rendered = _render(
        [
            {"role": "user", "content": "find shoes"},
            {
                "role": "assistant",
                "content": "I will search now.",
                "tool_calls": [
                    {"function": {"name": "search", "arguments": {"query": "shoes"}}}
                ],
            },
        ]
    )

    assert "I will search now.<|tool_call_start|>" in rendered


def test_lfm2_5_tool_role_output_remains_bare_chatml_content():
    rendered = _render(
        [
            {"role": "user", "content": "find shoes"},
            {
                "role": "assistant",
                "content": "<|tool_call_start|>[search()]<|tool_call_end|>",
            },
            {"role": "tool", "content": '{"ok": true}'},
        ]
    )

    assert "<|im_start|>tool\n" in rendered
    assert "<|tool_response_start|>" not in rendered
    assert "<|tool_response_end|>" not in rendered


def test_legacy_lfm2_tool_role_output_is_wrapped_by_template():
    rendered = _render(
        [
            {"role": "user", "content": "find shoes"},
            {
                "role": "assistant",
                "content": "<|tool_call_start|>[search()]<|tool_call_end|>",
            },
            {"role": "tool", "content": '{"ok": true}'},
        ],
        template_path=LEGACY_LFM2_TEMPLATE_PATH,
    )

    assert "<|im_start|>tool\n<|tool_response_start|>" in rendered
    assert "<|tool_response_end|><|im_end|>" in rendered


def test_legacy_lfm2_marks_assistant_generation_span():
    rendered, generation_indices = _render_with_assistant_indices(
        [
            {"role": "user", "content": "find shoes"},
            {
                "role": "assistant",
                "content": "<|tool_call_start|>[search()]<|tool_call_end|>",
            },
            {"role": "tool", "content": '{"ok": true}'},
        ],
        template_path=LEGACY_LFM2_TEMPLATE_PATH,
    )

    assert len(generation_indices) == 1
    start, end = generation_indices[0]
    assert rendered[start:end] == (
        "<|tool_call_start|>[search()]<|tool_call_end|><|im_end|>\n"
    )


def test_tools_are_serialized_with_family_specific_contracts():
    messages = [{"role": "user", "content": "find shoes"}]

    lfm2_5 = _render(messages, tools=SAMPLE_TOOLS)
    legacy_lfm2 = _render(
        messages,
        template_path=LEGACY_LFM2_TEMPLATE_PATH,
        tools=SAMPLE_TOOLS,
    )

    assert "List of tools: " in lfm2_5
    assert "<|tool_list_start|>" not in lfm2_5
    assert "<|tool_list_end|>" not in lfm2_5
    assert "<|tool_list_start|>" in legacy_lfm2
    assert "<|tool_list_end|>" in legacy_lfm2
