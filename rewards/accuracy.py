"""Math accuracy reward — re-exports trl.rewards.accuracy_reward.

Uses `math_verify` to parse and verify boxed LaTeX answers against a
ground-truth `solution` column. Returns 1.0 if the prediction matches,
0.0 if it doesn't, or None if the gold solution can't be parsed (in which
case GRPOTrainer skips this reward for that sample).

Requires:
    uv pip install math_verify

Dataset must have a `solution` column (string).

Example:
    >>> from rewards.accuracy import accuracy_reward
    >>> solutions = [r"\\frac{1}{3}"]
    >>> completions = [[{"role": "assistant", "content": r"My answer is \\boxed{\\frac{1}{3}}"}]]
    >>> accuracy_reward(completions, solutions)
    [1.0]
"""

from trl.rewards import accuracy_reward

__all__ = ["accuracy_reward"]
