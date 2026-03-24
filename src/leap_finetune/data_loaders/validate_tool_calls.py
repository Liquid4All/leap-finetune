import logging
from typing import Any

logger = logging.getLogger(__name__)

# === LFM tool call special tokens ===

_LFM_TOOL_CALL_START = "<|tool_call_start|>"
_LFM_TOOL_CALL_END = "<|tool_call_end|>"
_LFM_TOOL_RESPONSE_START = "<|tool_response_start|>"
_LFM_TOOL_RESPONSE_END = "<|tool_response_end|>"

# Foreign markers that indicate wrong-format tool call data
_FOREIGN_TOOL_MARKERS = [
    ("<tool_call>", "Qwen"),
    ("</tool_call>", "Qwen"),
    ("[TOOL_CALLS]", "Mistral"),
    ("[/TOOL_CALLS]", "Mistral"),
]


def has_foreign_tool_markers(text: str) -> str | None:
    """Check if text contains foreign (non-LFM) tool call markers.
    Returns the format name if found, None otherwise.
    """
    for marker, fmt in _FOREIGN_TOOL_MARKERS:
        if marker in text:
            return fmt
    return None


def validate_tool_calls_in_messages(messages: list, sample_idx: int) -> None:
    """Validate tool call formatting in a message list. Raises ValueError on issues.

    Checks:
    1. No foreign tool call markers (Qwen, Mistral)
    2. If pre-formatted LFM markers exist, tool call must come before text
    3. If assistant has tool call, a role="tool" response should follow
    4. Structured tool_calls field is not supported by LFM template
    """
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")

        # === Check for unsupported structured tool_calls field ===
        if role == "assistant" and "tool_calls" in msg:
            raise ValueError(
                f"Sample {sample_idx}, message {msg_idx}: 'tool_calls' field is not "
                f"supported by the LFM chat template. Tool calls must be pre-formatted "
                f"in the 'content' field using <|tool_call_start|>/<|tool_call_end|> markers."
            )

        # Only check content strings on assistant messages
        if role != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue

        # === Check for foreign markers ===
        fmt = has_foreign_tool_markers(content)
        if fmt:
            raise ValueError(
                f"Sample {sample_idx}, message {msg_idx}: Found {fmt} tool call markers "
                f"in assistant content. LFM requires '<|tool_call_start|>'/'<|tool_call_end|>' "
                f"markers in the content field."
            )

        # === Check ordering: tool call must come before text ===
        tc_pos = content.find(_LFM_TOOL_CALL_START)
        if tc_pos == -1:
            continue

        text_before = content[:tc_pos].strip()
        if text_before:
            raise ValueError(
                f"Sample {sample_idx}, message {msg_idx}: Text appears before tool call "
                f"in assistant content. LFM expects tool call first, then text. "
                f"Found: '{text_before[:50]}...' before <|tool_call_start|>"
            )

    # === Check tool call / tool response pairing ===
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")

        has_tool_call = isinstance(content, str) and _LFM_TOOL_CALL_START in content
        if not has_tool_call or role != "assistant":
            continue

        found_tool_response = False
        for next_msg in messages[msg_idx + 1 :]:
            if not isinstance(next_msg, dict):
                continue
            next_role = next_msg.get("role", "")
            if next_role == "tool":
                found_tool_response = True
                break
            if next_role == "user":
                break

        if not found_tool_response:
            raise ValueError(
                f"Sample {sample_idx}, message {msg_idx}: Assistant has tool call but "
                f"no tool response (role='tool') found before next user message."
            )


def validate_tool_calls_in_text(text: str, sample_idx: int, field: str) -> None:
    """Validate tool call formatting in a pre-formatted text string (DPO)."""
    fmt = has_foreign_tool_markers(text)
    if fmt:
        raise ValueError(
            f"Sample {sample_idx}, {field}: Found {fmt} tool call markers. "
            f"LFM requires '<|tool_call_start|>'/'<|tool_call_end|>' markers."
        )

    tc_pos = text.find(_LFM_TOOL_CALL_START)
    if tc_pos == -1:
        return

    text_before = text[:tc_pos].strip()
    if text_before:
        raise ValueError(
            f"Sample {sample_idx}, {field}: Text appears before tool call. "
            f"LFM expects tool call first, then text. "
            f"Found: '{text_before[:50]}...' before <|tool_call_start|>"
        )


def validate_tool_calls_dpo(chosen: Any, rejected: Any, sample_idx: int) -> None:
    """Validate tool call formatting in DPO chosen/rejected data."""
    for field, data in [("chosen", chosen), ("rejected", rejected)]:
        if isinstance(data, str):
            validate_tool_calls_in_text(data, sample_idx, field)
        elif isinstance(data, list):
            validate_tool_calls_in_messages(data, sample_idx)
