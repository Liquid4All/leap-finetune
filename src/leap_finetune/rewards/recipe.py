"""Recipe base class for GRPO task recipes.

A ``Recipe`` bundles together everything the reward side of a GRPO task
needs: a set of reward functions with weights, the dataset columns it
expects, a recommended system prompt, and a short description. The
training loop only consumes ``rewards()``; the other attributes are
documentation for the customer when they prep their dataset.

The whole extension story is **plain Python subclassing**. No registry,
no decorators, no config DSL — just override ``rewards()`` in a subclass
and return a new list of ``(callable, weight)`` pairs.

## Usage from YAML

    rewards:
      recipe: "./rewards/vlm_grounding.py::VLMGroundingRecipe"

## Writing a recipe

    from leap_finetune.rewards import Recipe

    def reward_a(completions, **kwargs):
        return [...]

    def reward_b(completions, bboxes_gt, **kwargs):
        return [...]

    class MyTaskRecipe(Recipe):
        description = "One-line summary of the task"
        required_columns = ("prompt", "bboxes_gt")
        system_prompt = "You are a ... Respond with ..."

        def rewards(self):
            return [
                (reward_a, 0.2),
                (reward_b, 1.0),
            ]

## Extending a shipped recipe

    from leap_finetune.rewards import load_recipe

    VLMGroundingRecipe = load_recipe(
        "./rewards/vlm_grounding.py::VLMGroundingRecipe"
    )

    def description_judge_reward(completions, object_descriptions, **kwargs):
        return [...]

    class GroundingWithCaptionsRecipe(VLMGroundingRecipe):
        required_columns = VLMGroundingRecipe.required_columns + ("object_descriptions",)

        def rewards(self):
            return [
                *super().rewards(),
                (description_judge_reward, 0.3),
            ]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar


class Recipe:
    """Base class for GRPO task recipes.

    Subclasses override :meth:`rewards` to return the list of reward
    functions and their weights for the task. The other class attributes
    are documentation: they describe what the recipe expects from the
    dataset and what prompt format the customer should use when preparing
    their data. The training loop does NOT auto-inject ``system_prompt``
    into the dataset — data preparation is the customer's concern.
    """

    #: Short human-readable description of the task. Shown in logs when the
    #: recipe is loaded.
    description: ClassVar[str] = ""

    #: Dataset columns the recipe's rewards expect (e.g. ``("prompt", "bboxes_gt")``).
    #: Documentation only — not automatically validated in v1.
    required_columns: ClassVar[tuple[str, ...]] = ()

    #: Recommended system prompt for this task. The customer should include
    #: it in their dataset prep — we do NOT auto-inject it.
    system_prompt: ClassVar[str | None] = None

    def rewards(self) -> list[tuple[Callable, float]]:
        """Return the list of ``(reward_function, weight)`` pairs for this task.

        Subclasses **must** override this. To extend a parent recipe,
        call ``super().rewards()`` and append:

        .. code-block:: python

            def rewards(self):
                return [
                    *super().rewards(),
                    (my_extra_reward, 0.3),
                ]
        """
        raise NotImplementedError(
            f"{type(self).__name__}.rewards() must be overridden and return a "
            "list of (callable, float) tuples."
        )
