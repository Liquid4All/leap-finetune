"""Recipe base class for GRPO task bundles.

A ``Recipe`` bundles reward functions with weights, plus documentation
for dataset prep (required columns, recommended system prompt). The
training loop only consumes ``rewards()``; the rest is guidance.

Extension is plain Python subclassing — override ``rewards()``.

Example::

    from leap_finetune.rewards import Recipe

    class MyTaskRecipe(Recipe):
        description = "One-line summary of the task"
        required_columns = ("prompt", "solution")
        system_prompt = "You are a ... Respond with ..."

        def rewards(self):
            return [(reward_a, 0.2), (reward_b, 1.0)]

Extending a shipped recipe::

    from leap_finetune.rewards import load_recipe

    VLMGroundingRecipe = load_recipe(
        "./rewards/tasks/vlm_grounding/recipe.py::VLMGroundingIoURecipe"
    )

    class GroundingWithCaptionsRecipe(VLMGroundingRecipe):
        required_columns = VLMGroundingRecipe.required_columns + ("object_descriptions",)

        def rewards(self):
            return [*super().rewards(), (description_judge_reward, 0.3)]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar


class Recipe:
    """Base class for GRPO task recipes.

    Override :meth:`rewards` to return the list of reward functions and
    their weights. The other class attributes document what the recipe
    expects from the dataset — they are not auto-validated and
    ``system_prompt`` is not auto-injected.
    """

    #: Short human-readable description. Shown in logs when loaded.
    description: ClassVar[str] = ""

    #: Dataset columns the rewards expect. Documentation only.
    required_columns: ClassVar[tuple[str, ...]] = ()

    #: Recommended system prompt. Include it in your dataset prep.
    system_prompt: ClassVar[str | None] = None

    def rewards(self) -> list[tuple[Callable, float]]:
        """Return ``[(reward_fn, weight), ...]`` for this task.

        Subclasses must override. To extend a parent, spread
        ``super().rewards()`` and append new pairs.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.rewards() must be overridden and return a "
            "list of (callable, float) tuples."
        )
