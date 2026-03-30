# Stress Test LFM2 vs LFM2.5 Tool Calling Templates

## Context

There are critical differences between LFM2 and LFM2.5 tool calling formats that affect training data validation and inference parsing. We need to empirically document and stress test these differences to ensure leap-finetune produces correct training data for each model variant.

### Model → Format mapping

- **LFM2 format**: All LFM2 models EXCEPT 24B (LFM2-1.2B, LFM2-2.6B, LFM2-8B-A1B, etc.)
- **LFM2.5 format**: All LFM2.5 models + LFM2-24B-A2B

### Known differences (verified by reading the actual Jinja templates)

Templates read from HF cache:

- LFM2: `~/.cache/huggingface/hub/models--LiquidAI--LFM2-1.2B/.../chat_template.jinja`
- LFM2.5: `~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/.../chat_template.jinja`

| Feature                           | LFM2                                                       | LFM2.5                                                           |
| --------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| **Tool list**                     | `<\|tool_list_start\|>[...]<\|tool_list_end\|>`            | Plain `[...]` (no special tokens)                                |
| **Tool call**                     | `<\|tool_call_start\|>[...]<\|tool_call_end\|>` (pythonic) | `<\|tool_call_start\|>[...]<\|tool_call_end\|>` (pythonic, same) |
| **Tool response**                 | `<\|tool_response_start\|>content<\|tool_response_end\|>`  | Plain content (no wrapping tokens)                               |
| **Thinking**                      | Not supported                                              | `</think>` stripping in history (`keep_past_thinking` param)     |
| **JSON tool calls**               | Pythonic only                                              | Pythonic by default, JSON optional via system prompt             |
| **Structured `tool_calls` field** | Ignored (crashes without content)                          | Ignored (same behavior)                                          |
| **Token special flag**            | `tool_call_start/end` are `special: true`                  | `tool_call_start/end` are `special: false`                       |

### Why this matters for training data

If training data for LFM2.5 includes `<|tool_response_start|>`/`<|tool_response_end|>` in tool messages, or `<|tool_list_start|>`/`<|tool_list_end|>` in system prompts, the model will see tokens it wasn't post-trained with. Vice versa for LFM2.

The `apply_chat_template()` function handles these differences automatically — but only if the data uses the structured message format (role="tool", tools= parameter). If training data is pre-formatted (tool markers already baked into content strings), the template passes it through verbatim and won't fix wrong markers.

### Parsing confusion across backends

Different inference backends parse tool calls differently:

- **sglang**: Has built-in `--tool-call-parser lfm2` support
- **vLLM**: No built-in LFM2 parser. Can try `pythonic` parser or write custom plugin via `--tool-parser-plugin`. Needs custom chat template.
- **llama.cpp**: Works if GGUF has correct chat template
- **Output format**: LFM2/2.5 use bracket notation `[get_weather(city="Antwerp")]` not XML `<tool_call>` tags. Parsers expecting XML will score 0 even when the model is correct.

## Test plan

### File: `tests/test_tool_call_templates.py` (expand existing)

Parametrize all tests across both tokenizers:

- `LFM2-1.2B` (LFM2 format)
- `LFM2.5-1.2B-Instruct` (LFM2.5 format)

### Shared test fixtures

```python
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

MULTI_TOOLS = [
    SAMPLE_TOOLS[0],
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]
```

### Test scenarios (all parametrized across both tokenizers)

#### 1. Single tool, single turn (tools= parameter)

```python
messages = [
    {"role": "user", "content": "What's the weather in SF?"},
]
output = tokenizer.apply_chat_template(messages, tools=SAMPLE_TOOLS, tokenize=False)
```

**Assert LFM2**: `<|tool_list_start|>` and `<|tool_list_end|>` in output
**Assert LFM2.5**: NO `<|tool_list_start|>`, just `List of tools: [` in output

#### 2. Multiple tools available

```python
messages = [
    {"role": "user", "content": "What's the weather?"},
]
output = tokenizer.apply_chat_template(messages, tools=MULTI_TOOLS, tokenize=False)
```

**Assert both**: All 3 tool names appear in output (`get_weather`, `search_web`, `send_email`)
**Assert LFM2**: Wrapped in `<|tool_list_start|>...<|tool_list_end|>`
**Assert LFM2.5**: Plain `[...]`

#### 3. Tool response formatting

```python
messages = [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": '<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>'},
    {"role": "tool", "content": '{"temp": 18, "unit": "celsius"}'},
    {"role": "assistant", "content": "It's 18°C in SF."},
]
output = tokenizer.apply_chat_template(messages, tokenize=False)
```

**Assert LFM2**: `<|tool_response_start|>` and `<|tool_response_end|>` wrap `{"temp": 18...}`
**Assert LFM2.5**: NO `<|tool_response_start|>`, just `<|im_start|>tool\n{"temp": 18...}<|im_end|>`

#### 4. Multi-turn with tool calls

```python
messages = [
    {"role": "user", "content": "What's the weather in SF?"},
    {"role": "assistant", "content": '<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>'},
    {"role": "tool", "content": '{"temp": 18}'},
    {"role": "assistant", "content": "It's 18°C."},
    {"role": "user", "content": "Now search for restaurants there."},
    {"role": "assistant", "content": '<|tool_call_start|>[search_web(query="restaurants in SF")]<|tool_call_end|>'},
    {"role": "tool", "content": '["Restaurant A", "Restaurant B"]'},
    {"role": "assistant", "content": "I found Restaurant A and B."},
]
output = tokenizer.apply_chat_template(messages, tokenize=False)
```

**Assert both**: Two `<|tool_call_start|>` occurrences
**Assert LFM2**: Two `<|tool_response_start|>` occurrences
**Assert LFM2.5**: Zero `<|tool_response_start|>` occurrences

#### 5. Pre-formatted content passthrough

```python
messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": '<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>\nChecking.'},
]
output = tokenizer.apply_chat_template(messages, tokenize=False)
```

**Assert both**: Exactly 1 `<|tool_call_start|>` (no double-wrapping)
**Assert both**: "Checking." appears after `<|tool_call_end|>`

#### 6. Tool call ordering (text + tool call)

```python
# Correct order: tool call first, then text
messages_correct = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": '<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>\nLet me check.'},
]

# Wrong order: text first, then tool call
messages_wrong = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": 'Let me check.\n<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>'},
]
```

**Assert both**: Both pass through verbatim (template doesn't reorder)
**Assert both**: Wrong order is NOT auto-corrected (this is what validation catches)

#### 7. Tool call with no arguments

```python
messages = [
    {"role": "user", "content": "What time is it?"},
    {"role": "assistant", "content": '<|tool_call_start|>[get_time()]<|tool_call_end|>'},
    {"role": "tool", "content": '{"time": "3:00 PM"}'},
]
output = tokenizer.apply_chat_template(messages, tokenize=False)
```

**Assert both**: `get_time()` in output
**Assert LFM2**: `<|tool_response_start|>` wraps response
**Assert LFM2.5**: No wrapping

#### 8. Empty tool response

```python
messages = [
    {"role": "user", "content": "Delete file"},
    {"role": "assistant", "content": '<|tool_call_start|>[delete_file(path="/tmp/x")]<|tool_call_end|>'},
    {"role": "tool", "content": ""},
]
output = tokenizer.apply_chat_template(messages, tokenize=False)
```

**Assert LFM2**: `<|tool_response_start|><|tool_response_end|>` (empty wrapped)
**Assert LFM2.5**: `<|im_start|>tool\n<|im_end|>` (empty bare)

#### 9. Multiple tool calls in single assistant message

```python
messages = [
    {"role": "user", "content": "Get weather and search for restaurants"},
    {"role": "assistant", "content": '<|tool_call_start|>[get_weather(location="SF"), search_web(query="restaurants SF")]<|tool_call_end|>'},
    {"role": "tool", "content": '{"temp": 18}'},
    {"role": "tool", "content": '["Restaurant A"]'},
]
output = tokenizer.apply_chat_template(messages, tokenize=False)
```

**Assert both**: Single `<|tool_call_start|>` with both calls inside
**Assert LFM2**: Two `<|tool_response_start|>` occurrences
**Assert LFM2.5**: Zero `<|tool_response_start|>` occurrences

#### 10. Tokenization difference (special: true vs special: false)

```python
# Tokenize and check if tool_call tokens are treated as special
text = '<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>'
tokens = tokenizer.encode(text, add_special_tokens=False)
decoded = tokenizer.decode(tokens)
```

**Assert LFM2**: `<|tool_call_start|>` tokenizes to single token (special=true)
**Assert LFM2.5**: Check if `<|tool_call_start|>` still tokenizes to single token despite special=false

## Validation status

Current `validate_tool_calls.py` is already model-agnostic — it only validates things common to both formats:

- `<|tool_call_start|>`/`<|tool_call_end|>` (both use these)
- Ordering (tool call before text — both require this)
- `role="tool"` pairing (both require this)
- Foreign marker rejection (applies to both)

It does NOT check for `<|tool_list_start|>` or `<|tool_response_start|>` — those are handled by `apply_chat_template` automatically per model. The constants `_LFM_TOOL_RESPONSE_START`/`_LFM_TOOL_RESPONSE_END` in validate_tool_calls.py are dead code and should be cleaned up.

### Potential validation updates after testing

Depending on test results, we may need to:

- Remove unused constants (`_LFM_TOOL_RESPONSE_START`, etc.)
- Add a model family detection utility (`is_lfm25_format(model_name)`) for any model-specific checks
- Warn if pre-formatted data contains `<|tool_list_start|>` when training an LFM2.5 model (or vice versa)

## Files to modify

- `tests/test_tool_call_templates.py` — expand with parametrized tests across both tokenizers, all scenarios above
- `src/leap_finetune/data_loaders/validate_tool_calls.py` — clean up dead constants after testing confirms they're unused

## Verification

```bash
uv run pytest tests/test_tool_call_templates.py -v
```

Needs compute node for LFM2.5 tokenizer download (LFM2-1.2B is already cached locally).
