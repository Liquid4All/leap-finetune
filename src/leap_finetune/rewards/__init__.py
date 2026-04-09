"""Reward loading for GRPO training.

Customer-facing rewards live in the top-level ``rewards/`` directory of the
repo (sibling of ``job_configs/``). Individual reward functions are plain
Python files; task-level reward collections are :class:`Recipe` subclasses.
See ``rewards/README.md`` for the full guide.

Public API:

* :class:`Recipe` — base class that shipped and customer recipes subclass.
* :func:`load_recipe` — helper for customer recipe files that want to
  subclass a shipped recipe living in a sibling file.
* :func:`resolve_reward_specs` — turns a YAML ``rewards:`` entry into a
  list of callables + optional weights (used internally by the training
  loops; customers normally don't call it directly).
"""

from leap_finetune.rewards.loader import load_recipe, resolve_reward_specs
from leap_finetune.rewards.recipe import Recipe

__all__ = ["Recipe", "load_recipe", "resolve_reward_specs"]
