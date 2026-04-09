"""Visual grounding format reward — checks for `<bbox>x1,y1,x2,y2</bbox>`.

Pairs with `grounding_iou.py` and the `grounding_bbox` RL environment.
Returns 1.0 if exactly one well-formed bbox tag is present, 0.0 otherwise.
"""

import re

# Matches <bbox>123,456,789,1011</bbox>. Coordinates can be int or float.
BBOX_RE = re.compile(r"<bbox>\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*</bbox>")


def grounding_format_reward(completions, **kwargs) -> list[float]:
    """1.0 if completion contains exactly one <bbox>...</bbox> tag, 0.0 otherwise."""
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[0].get("content", "")
        else:
            text = str(completion)
        matches = BBOX_RE.findall(text)
        rewards.append(1.0 if len(matches) == 1 else 0.0)
    return rewards
