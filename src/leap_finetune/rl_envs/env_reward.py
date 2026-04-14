"""Reward function that forwards per-completion rewards from an OpenEnv rollout.

The adapter's rollout_func returns an ``env_reward`` field on the rollout
dict; TRL passes it through as a kwarg, and this function reads it back.
The training loop auto-prepends ``env_reward`` to the reward list when
``rl_env`` is set in YAML.
"""

from __future__ import annotations


def env_reward(completions, **kwargs) -> list[float]:
    env_rewards = kwargs.get("env_reward")
    if env_rewards is None:
        return [0.0] * len(completions)
    return [float(r) if r is not None else 0.0 for r in env_rewards]
