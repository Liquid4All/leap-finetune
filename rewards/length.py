# === Length shaping reward ===
#
# Scale completion length to [0, 1]. Copy this file as a template when a task
# needs simple shaping alongside a stronger correctness reward.

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
