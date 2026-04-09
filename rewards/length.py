"""Length-based shaping reward.

Scales completion length to [0, 1] using a target length. Useful for
encouraging the model to produce completions of a desired length during
early GRPO training (combine with weight ~0.1 alongside a stronger
correctness reward).

Edit `TARGET_LEN` to change the target, or copy this file as a template
to write your own variant.
"""

# Target completion length in characters. Completions at this length or
# longer get a reward of 1.0; shorter ones are linearly scaled down.
TARGET_LEN = 200


def length_reward(completions, **kwargs) -> list[float]:
    """Reward in [0, 1] proportional to completion length, capped at TARGET_LEN."""
    rewards = []
    for completion in completions:
        # Handle both conversational ([{"role": ..., "content": ...}]) and
        # string completions.
        if isinstance(completion, list):
            text = completion[0].get("content", "")
        else:
            text = str(completion)
        score = min(len(text) / TARGET_LEN, 1.0)
        rewards.append(float(score))
    return rewards
