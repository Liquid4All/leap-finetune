"""OpenEnv adapter for GRPO training.

Thin glue between TRL's ``rollout_func`` hook and the OpenEnv environment
standard (Meta-PyTorch + HuggingFace). Environments live on the HuggingFace
Hub as Spaces; we do NOT maintain a registry here.

See `src/leap_finetune/rl_envs/README.md` for the full guide.
"""

from leap_finetune.rl_envs.adapter import (
    build_openenv_rollout_func,
    connect_openenv,
)
from leap_finetune.rl_envs.env_reward import env_reward

__all__ = [
    "build_openenv_rollout_func",
    "connect_openenv",
    "env_reward",
]
