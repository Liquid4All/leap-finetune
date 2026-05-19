import ast
from pathlib import Path

from transformers.utils.chat_template_utils import render_jinja_template


LFM25_TEMPLATE_PATH = Path(
    "job_configs/chat_templates/lfm25_tool_call_chat_template.jinja"
)
LFM24B_TEMPLATE_PATH = Path(
    "job_configs/chat_templates/lfm2_24b_tool_call_chat_template.jinja"
)
LEGACY_LFM2_TEMPLATE_PATH = Path(
    "job_configs/chat_templates/lfm2_tool_call_chat_template.jinja"
)

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


def _render(messages, template_path=LFM25_TEMPLATE_PATH, tools=None):
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


def test_lfm25_renders_assistant_content_before_structured_tool_calls():
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


def test_lfm25_tool_role_output_remains_bare_chatml_content():
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


def test_tools_are_serialized_with_family_specific_contracts():
    messages = [{"role": "user", "content": "find shoes"}]

    lfm25 = _render(messages, tools=SAMPLE_TOOLS)
    legacy_lfm2 = _render(
        messages,
        template_path=LEGACY_LFM2_TEMPLATE_PATH,
        tools=SAMPLE_TOOLS,
    )

    assert "List of tools: " in lfm25
    assert "<|tool_list_start|>" not in lfm25
    assert "<|tool_list_end|>" not in lfm25
    assert "<|tool_list_start|>" in legacy_lfm2
    assert "<|tool_list_end|>" in legacy_lfm2


def test_24b_template_matches_canonical_lfm25_tool_contract():
    messages = [
        {"role": "user", "content": "find shoes"},
        {
            "role": "assistant",
            "content": "<|tool_call_start|>[search()]<|tool_call_end|>",
        },
        {"role": "tool", "content": '{"ok": true}'},
    ]

    canonical = _render(messages, template_path=LFM25_TEMPLATE_PATH, tools=SAMPLE_TOOLS)
    active_24b = _render(
        messages, template_path=LFM24B_TEMPLATE_PATH, tools=SAMPLE_TOOLS
    )

    assert active_24b == canonical
