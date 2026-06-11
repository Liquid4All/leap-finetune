---
name: lft-data
description: Prepare and validate leap-finetune training data, eval data, tool-calling data, and multimodal datasets. Use when editing dataset loaders, formats, validation, tokenization, or data fixtures.
---

# LFT Data

## Use When

- Preparing SFT, DPO, GRPO, VLM, or eval datasets.
- Validating messages, tool calls, image paths, or preference rows.
- Editing dataset loading, tokenization, or Ray Data utilities.
- Creating small fixtures for tests.

## Do Not Use When

- Designing proprietary hosted data-generation recipes. Keep those outside this
  OSS repo unless the user explicitly asks for public primitives.
- Adding benchmark metrics only. Use `lft-evals`.

## Canonical Files

- `README.md`
- `src/leap_finetune/data_loading/`
- `src/leap_finetune/data_loading/validate_tool_format.py`
- `tests/test_data.py`
- `tests/test_tool_call_validation.py`
- `tests/e2e/fixtures/`

## Steps

1. Identify the data contract: SFT messages, DPO preference rows, GRPO
   prompt/solution, VLM messages, tool calling, or benchmark eval rows.
2. Keep new fixtures tiny and deterministic.
3. Prefer JSONL, JSON, Parquet, or CSV paths supported by the loader.
4. For VLM data, resolve relative images through `dataset.image_root`.
5. For tool calling, use LFM bracket notation in assistant `content` or rely on
   supported OpenAI `tool_calls` conversion.
6. Add validation tests when changing accepted or rejected formats.

## Expected Output

- A dataset format recommendation, fixture, validation change, or loader patch.
- A minimal example row when the data contract is the user-facing artifact.
- The exact validation or data test command.
- A note on where large generated data should live outside git.

## Verification

```bash
uv run pytest tests/test_data.py tests/test_tool_call_validation.py tests/test_tool_call_templates.py
```

For loader or Ray Data changes:

```bash
uv run pytest tests/test_ray_data_utils.py tests/config
```

## Common Failures

- Foreign tool-call XML formats should be rejected with actionable errors.
- Do not add `<|tool_response_start|>` markers inside `role: "tool"` messages.
- DPO rows should use explicit `prompt`, `chosen`, and `rejected` when possible.
- Large generated datasets belong in external storage, not git.
