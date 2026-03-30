import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# === Marker constants ===

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_LIST_START = "<|tool_list_start|>"
TOOL_LIST_END = "<|tool_list_end|>"
TOOL_RESPONSE_START = "<|tool_response_start|>"
TOOL_RESPONSE_END = "<|tool_response_end|>"

FOREIGN_MARKERS = [
    "<tool_call>",
    "</tool_call>",
    "<function_call>",
    "</function_call>",
    "<|plugin|>",
    "<start_function_call>",
    "<end_function_call>",
]


# === Detection ===


@dataclass
class ToolFormatInfo:
    has_tool_calls: bool = False
    has_tool_call_markers: bool = False
    has_tool_list_markers: bool = False
    has_tool_response_markers: bool = False
    has_tool_role: bool = False
    has_structured_tool_calls: bool = False
    has_foreign_markers: bool = False
    foreign_markers_found: list[str] = field(default_factory=list)
    detected_format: str = "none"


def detect_tool_format(samples: list[dict]) -> ToolFormatInfo:
    """Scan a list of samples for tool call markers and return a format descriptor.

    Each sample should have a 'messages' or 'conversations' key containing a
    list of message dicts with 'role' and 'content'.
    """
    info = ToolFormatInfo()
    foreign_set = set()

    for sample in samples:
        messages = sample.get("messages") or sample.get("conversations", [])
        if not messages:
            continue

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""

            if role == "tool":
                info.has_tool_role = True

            if TOOL_CALL_START in content:
                info.has_tool_call_markers = True
            if TOOL_LIST_START in content:
                info.has_tool_list_markers = True
            if TOOL_RESPONSE_START in content:
                info.has_tool_response_markers = True

            if role == "assistant" and "tool_calls" in msg:
                tool_calls = msg["tool_calls"]
                if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                    info.has_structured_tool_calls = True

            for marker in FOREIGN_MARKERS:
                if marker in content:
                    foreign_set.add(marker)

    info.foreign_markers_found = sorted(foreign_set)
    info.has_foreign_markers = len(foreign_set) > 0
    info.has_tool_calls = (
        info.has_tool_call_markers
        or info.has_tool_role
        or info.has_structured_tool_calls
        or info.has_foreign_markers
    )

    # Determine format
    if info.has_foreign_markers:
        info.detected_format = "foreign"
    elif info.has_structured_tool_calls:
        info.detected_format = "structured"
    elif info.has_tool_list_markers and info.has_tool_call_markers:
        info.detected_format = "lfm2"
    elif info.has_tool_call_markers:
        info.detected_format = "lfm25"
    else:
        info.detected_format = "none"

    return info


# === Validation ===


@dataclass
class ToolFormatIssue:
    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    auto_fixable: bool = False
    fix_hint: str = ""


def validate_tool_format(
    format_info: ToolFormatInfo,
    model_family: str,
) -> list[ToolFormatIssue]:
    """Check format compatibility with target model. Return issues sorted by severity."""
    issues = []

    if not format_info.has_tool_calls:
        return issues

    if format_info.has_foreign_markers:
        markers = ", ".join(format_info.foreign_markers_found)
        issues.append(
            ToolFormatIssue(
                "error",
                "FOREIGN_MARKERS",
                f"Found foreign tool call markers: {markers}. Not LFM format.",
                fix_hint=f'Convert to LFM bracket notation: {TOOL_CALL_START}[func(arg="value")]{TOOL_CALL_END}',
            )
        )

    if format_info.has_structured_tool_calls:
        issues.append(
            ToolFormatIssue(
                "warning",
                "STRUCTURED_TOOL_CALLS",
                "Found 'tool_calls' field on assistant messages. LFM templates drop this. Will auto-convert to bracket notation.",
                auto_fixable=True,
            )
        )

    if format_info.has_tool_response_markers:
        issues.append(
            ToolFormatIssue(
                "info",
                "PREBAKED_TOOL_RESPONSE",
                f"Stripping pre-baked {TOOL_RESPONSE_START} from role=tool to prevent double wrapping.",
                auto_fixable=True,
            )
        )

    if format_info.has_tool_list_markers and model_family == "lfm25":
        issues.append(
            ToolFormatIssue(
                "warning",
                "TOOL_LIST_MARKERS_ON_LFM25",
                f"Found {TOOL_LIST_START} markers not used by LFM2.5. Will strip.",
                auto_fixable=True,
            )
        )

    if (
        format_info.has_tool_call_markers
        and not format_info.has_tool_list_markers
        and model_family == "lfm2"
    ):
        issues.append(
            ToolFormatIssue(
                "warning",
                "MISSING_TOOL_LIST_MARKERS",
                f"Missing {TOOL_LIST_START} markers expected by LFM2. Training will work but won't fully match pretraining.",
                fix_hint=f"Wrap tool defs with {TOOL_LIST_START}[...]{TOOL_LIST_END} or use tools= parameter.",
            )
        )

    if format_info.detected_format == model_family:
        issues.append(
            ToolFormatIssue(
                "info",
                "FORMAT_MATCHED",
                f"Tool call format matches {model_family.upper()} model.",
            )
        )

    return sorted(
        issues, key=lambda i: {"error": 0, "warning": 1, "info": 2}[i.severity]
    )


# === Normalization (per-row, for Ray .map()) ===


def _tool_calls_to_pythonic(tool_calls: list[dict]) -> str:
    """Convert structured tool_calls list to LFM pythonic bracket notation."""
    calls = []
    for tc in tool_calls:
        func = tc.get("function", tc)
        name = func["name"]
        args = func.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)

        parts = []
        for k, v in args.items():
            if isinstance(v, str):
                parts.append(f'{k}="{v}"')
            else:
                parts.append(f"{k}={v}")
        calls.append(f"{name}({', '.join(parts)})")

    return f"{TOOL_CALL_START}[{', '.join(calls)}]{TOOL_CALL_END}"


def normalize_tool_format(row: dict, model_family: str) -> dict:
    """Auto-fix tool call format mismatches. Stateless, used as Ray .map() fn."""
    messages = row.get("messages") or row.get("conversations")
    if not messages:
        return row

    modified = False
    new_messages = []

    for msg in messages:
        new_msg = dict(msg)
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        # 1. Strip tool_list markers for LFM2.5
        if role == "system" and model_family == "lfm25":
            if TOOL_LIST_START in content or TOOL_LIST_END in content:
                content = content.replace(TOOL_LIST_START, "")
                content = content.replace(TOOL_LIST_END, "")
                new_msg["content"] = content
                modified = True

        # 2. Strip pre-baked tool_response markers (prevents double wrapping)
        if role == "tool":
            if TOOL_RESPONSE_START in content or TOOL_RESPONSE_END in content:
                content = content.replace(TOOL_RESPONSE_START, "")
                content = content.replace(TOOL_RESPONSE_END, "")
                new_msg["content"] = content
                modified = True

        # 3. Convert structured tool_calls to pre-baked content
        if role == "assistant" and "tool_calls" in msg:
            tool_calls = msg["tool_calls"]
            if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                pythonic = _tool_calls_to_pythonic(tool_calls)
                existing_content = content.strip()
                if existing_content:
                    new_msg["content"] = f"{pythonic}\n{existing_content}"
                else:
                    new_msg["content"] = pythonic
                # Remove the tool_calls field so template doesn't try to process it
                new_msg.pop("tool_calls", None)
                modified = True

        new_messages.append(new_msg)

    if not modified:
        return row

    # Preserve the original column name
    col_name = "messages" if "messages" in row else "conversations"
    new_row = dict(row)
    new_row[col_name] = new_messages
    return new_row


def get_tool_normalizer(model_family: str):
    """Return a Ray-compatible map function for tool call normalization."""

    def _normalize(row: dict) -> dict:
        return normalize_tool_format(row, model_family)

    return _normalize
