# Tool-Call Template Cleanup Plan

## Current Discovery

- Tool-call normalization already converts structured assistant `tool_calls` into LFM pythonic content and strips the `tool_calls` field before tokenization.
- The current custom Shopify chat template also renders structured `message.tool_calls`, which duplicates preprocessing responsibility and can make training/inference behavior diverge.
- `LiquidAI/LFM2-24B-A2B` uses LFM2.5-style tool formatting: plain tool definitions in the system prompt as `List of tools: [...]`, assistant calls wrapped with `<|tool_call_start|>` / `<|tool_call_end|>`, and no legacy tool-list/tool-response sentinels.
- Older LFM2-style examples use `<|tool_list_start|>` / `<|tool_list_end|>` and `<|tool_response_start|>` / `<|tool_response_end|>`, so README/tests should explicitly distinguish LFM2 vs LFM2.5/24B behavior.
- The Shopify filtered dataset was built with a different external template than the current training template, which is why rows that were previously under the length cap can exceed the current rendered/tokenized cap.

## Intended Boundary

- Preprocessing owns data normalization:
  - Convert structured assistant `tool_calls` into pre-baked LFM pythonic content.
  - Strip `tool_calls` after conversion.
  - Handle dict arguments, JSON-string arguments, empty arguments, and `None` arguments.
  - Reject foreign tool-call formats with actionable validation errors.
  - Strip legacy markers when training LFM2.5/24B-family models.
- Chat templates own message serialization only:
  - Render the system message.
  - Render `tools=` into the model-family-specific system-prompt format.
  - Render message roles/content into ChatML.
  - Do not repair, reorder, or synthesize assistant tool calls from structured `tool_calls`.

## Implementation Plan

- Replace the custom Shopify template behavior with LFM2.5/24B-compatible behavior:
  - Keep system-message rendering.
  - Render `tools=` as plain `List of tools: [...]`.
  - Remove Jinja-side structured `message.tool_calls` rendering.
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
