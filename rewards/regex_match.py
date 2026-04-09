"""Generic regex-match reward — template for ad-hoc format rewards.

Returns 1.0 if the completion matches PATTERN, 0.0 otherwise. Copy this
file and edit PATTERN for your specific format. For more complex format
checking, use a custom Python function instead of regex.
"""

import re

# Edit this to your format. Examples:
#   r"^Answer: \w+$"                    — single-word answer after "Answer:"
#   r"\b\d{4}-\d{2}-\d{2}\b"            — ISO date
#   r"```python\n.*?\n```"              — Python code block
PATTERN = re.compile(r"^Answer: .+$", re.MULTILINE)


def regex_match_reward(completions, **kwargs) -> list[float]:
    """1.0 if PATTERN matches the completion, 0.0 otherwise."""
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[0].get("content", "")
        else:
            text = str(completion)
        rewards.append(1.0 if PATTERN.search(text) else 0.0)
    return rewards
