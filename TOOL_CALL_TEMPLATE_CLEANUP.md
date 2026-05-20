# Tool-Call Template Cleanup Plan

## Current Discovery

- Tool-call normalization already converts structured assistant `tool_calls` into LFM pythonic content and strips the `tool_calls` field before tokenization.
- The current custom Shopify chat template also renders structured `message.tool_calls`, which duplicates preprocessing responsibility and can make training/inference behavior diverge.
- `LiquidAI/LFM2-24B-A2B` uses LFM2.5-style tool formatting: plain tool definitions in the system prompt as `List of tools: [...]`, assistant calls wrapped with `<|tool_call_start|>` / `<|tool_call_end|>`, and no legacy tool-list/tool-response sentinels.
- Older LFM2-style examples use `<|tool_list_start|>` / `<|tool_list_end|>` and `<|tool_response_start|>` / `<|tool_response_end|>`, so README/tests should explicitly distinguish LFM2 vs LFM2.5/24B behavior.
- The Shopify filtered dataset was built with a different external template than the current training template, which is why rows that were previously under the length cap can exceed the current rendered/tokenized cap.

## 2026-05-18 Verification

- The active Shopify template and the referenced official LFM2.5-style template both had the same string-argument escaping issue: `query="women's shoes"` rendered as invalid pythonic syntax when arguments arrived as a dict.
- The active template crashed on standard TRL/HF assistant tool-call rows with `content: null`; the downloaded Shopify parquet materializes these as empty strings, but the null guard is still required for compatibility.
- The active template only accepted nested OpenAI tool calls. The downloaded Shopify parquet is nested-only, but the loader/validator already supports flat `{name, arguments}` calls, so the template now accepts both.
- Full Shopify dataset scan under `/lambdafs/alay/datasets/Shopify_sidekick-suggested-actions-distillation`: 21,843 rows, 1,843,226 messages, 521,247 assistant messages, and 1,256,450 bare `role="tool"` messages.
- Assistant tool-call arguments are JSON strings in the parquet schema. Parsed values frequently contain special characters: 102,048 apostrophes, 14,895 quotes, 11,539 newlines, and 3,036 backslashes.
- The official LFM2.5-style template does not wrap `role="tool"` content in `<|tool_response_start|>` / `<|tool_response_end|>`, and the Shopify data itself stores bare tool responses. Do not add legacy response wrappers to the 24B Shopify template unless the posttraining owner confirms the official template contract should change.
- Sampled 1,000 rows each from `LiquidAI/liquid_format_tool_ace_justin-scored` and `LiquidAI/FunReason-MT-nothink-processed-32k`: all sampled `role="tool"` messages were bare, and assistant tool-call messages were tool-call-only (`<|tool_call_start|>...<|tool_call_end|>`) with no prose before/after.
- Rendering the same real FunReason row through templates showed the family split: `LFM2-2.6B-Exp` legacy template inserts tool-response wrappers; local `lfm2.5-VL-1.6B`, the official 8B MoE template, and the active 24B Shopify template do not.
- Small dense model check with `LiquidAI/LFM2.5-1.2B-Instruct`: after the prefix `<|im_start|>tool\n`, the raw tool-result `[` token was preferred over `<|tool_response_start|>` by 10.3 nats. That makes wrappers off-distribution for LFM2.5/24B.
- Full Shopify scan found 277,044 assistant messages with both non-empty `content` and structured `tool_calls`; for LFM2.5/24B these must render as content first, then `<|tool_call_start|>...`, matching the official template. Legacy LFM2 keeps tool-call-first handling for backward compatibility.

## Intended Boundary

- Preprocessing owns data normalization:
  - Convert structured assistant `tool_calls` into pre-baked LFM pythonic content.
  - Strip `tool_calls` after conversion.
  - Preserve model-family ordering when assistant prose and structured `tool_calls` coexist: legacy LFM2 writes tool-call first; LFM2.5/24B writes prose first and tool-call second.
  - Handle dict arguments, JSON-string arguments, empty arguments, and `None` arguments.
  - Parse JSON-string `function.arguments` into named pythonic arguments before any chat template renders the row.
  - Reject foreign tool-call formats with actionable validation errors.
  - Strip legacy markers when training LFM2.5/24B-family models.
- Chat templates own message serialization only:
  - Render the system message.
  - Render `tools=` into the model-family-specific system-prompt format.
  - Render message roles/content into ChatML.
  - Do not repair, reorder, or synthesize assistant tool calls from dirty structured `tool_calls`.
  - Structured `message.tool_calls` support is a compatibility fallback for eval/serving histories that already use dict arguments.
  - JSON-string `function.arguments` in structured history must be normalized before templating. The Jinja templates intentionally do not duplicate the loader's JSON parsing/validation logic.
- Canonical tracked templates:
- `job_configs/chat_templates/lfm2_tool_call_chat_template.jinja`: legacy LFM2, tool definitions wrapped with `<|tool_list_start|>` / `<|tool_list_end|>`, and `role="tool"` content wrapped with `<|tool_response_start|>` / `<|tool_response_end|>` during rendering.
- `job_configs/chat_templates/lfm25_tool_call_chat_template.jinja`: LFM2.5/24B, plain `List of tools: ...`, assistant calls as `<|tool_call_start|>...<|tool_call_end|>`, and bare ChatML `role="tool"` content.

## Implemented Fixes

- The active 24B Shopify template is kept equivalent to the tracked LFM2.5 canonical template.
- `job_configs/chat_templates/lfm2_tool_call_chat_template.jinja` tracks the legacy LFM2 contract.
- `job_configs/chat_templates/lfm25_tool_call_chat_template.jinja` tracks the LFM2.5/24B contract.
- Template-side structured `tool_calls` fallback now escapes string argument values, accepts null assistant content, and accepts flat or nested tool-call schemas.
- Template-side structured `tool_calls` fallback does not parse JSON-string arguments; preprocessing remains the canonical path for Shopify/OpenAI-style JSON-string arguments.
- Direct chat-template call sites normalize structured JSON-string arguments before rendering via `normalize_messages_for_chat_template()` / `normalize_row_for_chat_template()`.
- Preprocessing now preserves family-specific assistant prose/tool-call order when converting structured tool calls into content.
- Validation is family-aware: legacy LFM2 rejects text before pre-baked tool-call markers, while LFM2.5/24B allows it.

## Remaining Plan

- Replace the custom Shopify template behavior with LFM2.5/24B-compatible behavior:
  - Keep system-message rendering.
  - Render `tools=` as plain `List of tools: [...]`.
  - Keep Jinja-side structured `message.tool_calls` rendering only as a serving/eval compatibility fallback until all training and inference paths use one normalized representation.
  - Remove hard-coded dynamic date injection.
  - Do not emit `<|tool_list_start|>`, `<|tool_list_end|>`, `<|tool_response_start|>`, or `<|tool_response_end|>` for `LFM2-24B-A2B`.
- Prefer no template override for `LFM2-24B-A2B` if the base tokenizer already carries the correct official chat template; otherwise use a cleaned explicit template.
- Ensure HF checkpoint save/export persists the exact tokenizer chat template used for training and vLLM inference.
- Update vLLM eval scripts to stop depending on the old custom template, or point them at the cleaned LFM2.5/24B template when an explicit `--chat-template` is required.
- Keep `drop_overlength: true`; the final tokenized cache must be rebuilt under the exact final training render path.

## Tests

- Update tool-template tests to separate model families:
  - LFM2.5/24B: plain tool list in system prompt, plain `tool` role, assistant tool calls only from content.
  - Legacy LFM2: tool-list/tool-response marker behavior remains documented where supported.
- Add a normalization-plus-template smoke test:
  - Start with Shopify/OpenAI-style structured `tool_calls`.
  - Normalize the row.
  - Apply the LFM2.5/24B template.
  - Assert exactly one `<|tool_call_start|>` and one `<|tool_call_end|>`.
  - Assert no structured `tool_calls` are consumed by the template.
  - Assert no legacy tool-list/tool-response markers appear.
- Re-run:
  - `uv run --frozen pytest tests/test_tool_call_validation.py`
  - `uv run --frozen pytest tests/test_tool_call_templates.py`
  - `uv run --frozen pytest tests/test_sft_loss_masks.py`
  - `uv run --frozen pytest tests/test_ray_data_utils.py`

## Operational Note

Runs launched with the existing custom template are useful for CP/runtime signal, but should not be treated as final cleaned tool-call/template validation runs.
