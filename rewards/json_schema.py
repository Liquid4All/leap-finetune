"""JSON-output reward — checks the completion is valid JSON matching a schema.

Returns 1.0 if the completion parses as JSON and validates against SCHEMA,
0.0 otherwise. Edit SCHEMA for your task or copy this file as a template.

Useful for training models to output structured responses (tool calls,
extraction tasks, etc.).
"""

import json

# JSONSchema-style spec. Customize for your task.
SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
}


def _matches_schema(value, schema) -> bool:
    """Minimal JSONSchema-ish validator. Replace with `jsonschema` if needed."""
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return False
        for required_key in schema.get("required", []):
            if required_key not in value:
                return False
        for key, sub_schema in schema.get("properties", {}).items():
            if key in value and not _matches_schema(value[key], sub_schema):
                return False
        return True
    if expected == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items")
        if item_schema is None:
            return True
        return all(_matches_schema(v, item_schema) for v in value)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True  # unknown type — accept


def json_schema_reward(completions, **kwargs) -> list[float]:
    """1.0 if completion is valid JSON matching SCHEMA, 0.0 otherwise."""
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[0].get("content", "")
        else:
            text = str(completion)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            rewards.append(0.0)
            continue
        rewards.append(1.0 if _matches_schema(parsed, SCHEMA) else 0.0)
    return rewards
