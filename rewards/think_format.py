"""R1-style format reward — re-exports trl.rewards.think_format_reward.

Checks that the completion is wrapped in <think>...</think> tags. Returns
1.0 if the format matches, 0.0 otherwise. No dataset columns required.

Example:
    >>> from rewards.think_format import think_format_reward
    >>> completions = [
    ...     [{"role": "assistant", "content": "<think>reasoning</think>answer"}],
    ...     [{"role": "assistant", "content": "answer"}],
    ... ]
    >>> think_format_reward(completions)
    [1.0, 0.0]
"""

from trl.rewards import think_format_reward

__all__ = ["think_format_reward"]
