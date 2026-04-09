# Reward functions

This directory holds **reward functions** used by GRPO training. The same
mechanism works for everything: built-in rewards we ship and custom rewards
you write are both plain Python functions in this directory, referenced from
your YAML config by `path::function_name`.

## Using a shipped reward

```yaml
# job_configs/my_grpo.yaml
rewards:
  funcs:
    - "./rewards/accuracy.py::accuracy_reward"
    - "./rewards/think_format.py::think_format_reward"
  weights: [1.0, 0.2]   # optional; defaults to 1.0 for each
```

The `path` is resolved relative to the directory containing your YAML file
(or you can use an absolute path). The `function_name` is the name of any
top-level function in that file with the GRPO reward signature.

## Writing a custom reward

Drop a new `.py` file in this directory (or anywhere — the path in YAML can
be relative to the config or absolute) and reference it by path. No
decorators, no registry — just write a function:

```python
# rewards/my_custom.py
def my_custom_reward(completions, **kwargs):
    """Reward 1.0 for completions over 50 characters, 0.0 otherwise."""
    contents = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return [1.0 if len(t) >= 50 else 0.0 for t in contents]
```

```yaml
rewards:
  funcs:
    - "./rewards/my_custom.py::my_custom_reward"
```

## The reward function signature

```python
def reward_fn(completions, **kwargs) -> list[float | None]:
    ...
```

- **`completions`** — a list with one entry per generated completion.
  - For **conversational** prompts (chat models, the common case),
    each entry is `[{"role": "assistant", "content": "<text>"}]`.
  - For **string** prompts, each entry is the raw completion string.
  - Always extract content defensively: `c[0]["content"] if isinstance(c, list) else c`.
- **`**kwargs`** — every other column in your dataset row is forwarded as a
  keyword argument with the same name. For example, if your dataset has a
  `solution` column, your reward function receives `solution: list[str]`.
  TRL also forwards `prompts`, `completion_ids`, `trainer_state`, and (when
  using `rl_env`) `env_reward` from the rollout output. Use `**kwargs` so
  unused fields are silently ignored.
- **Return** — a list of floats, one per completion. You can return `None`
  for a sample to mark it as not applicable to this reward function (useful
  for multi-task datasets where different reward functions apply to
  different rows). TRL skips `None` entries during reward aggregation.

## Shipped rewards (individual functions)

| File | Function | What it does | Required dataset columns |
|------|----------|--------------|--------------------------|
| `accuracy.py` | `accuracy_reward` | Math accuracy via `math_verify` (re-export of `trl.rewards.accuracy_reward`). Returns 1.0 / 0.0 / `None`. | `solution` (str) |
| `think_format.py` | `think_format_reward` | Checks the completion is wrapped in `<think>...</think>` (re-export of `trl.rewards.think_format_reward`). | none |
| `length.py` | `length_reward` | Length-based shaping reward, scaled to `[0, 1]`. Useful for encouraging longer (or shorter) completions during early training. | none |
| `json_schema.py` | `json_schema_reward` | Parses the completion as JSON and validates it against a hardcoded schema. Edit the schema in the file or copy as a template. | none |
| `regex_match.py` | `regex_match_reward` | Generic regex match against a hardcoded pattern. Edit or copy as template. | none |
| `grounding_iou.py` | `grounding_iou_reward` | Legacy single-box IoU reader for `<bbox>...</bbox>` tagged output (pure IoU, no center-distance term). Superseded by `vlm_grounding.py::ciou_reward` for most use cases. | `bbox_gt` (list of 4 floats) |
| `grounding_format.py` | `grounding_format_reward` | Legacy `<bbox>x,y,x,y</bbox>` format check. | none |

## Shipped recipes

A **recipe** is a Python class that collects everything the reward side of
a GRPO task needs — the set of reward functions, their weights, the
dataset columns they expect, and a recommended system prompt — in one
file. Reference the class by `<path>::<ClassName>` from YAML and the
training loop pulls in all of the recipe's rewards automatically.

```yaml
rewards:
  recipe: "./rewards/vlm_grounding.py::VLMGroundingRecipe"
  # Optional — override the recipe's default weights:
  # weights: [0.1, 0.1, 1.0, 1.0]
  # Optional — stack extra individual rewards after the recipe:
  # funcs:
  #   - "./rewards/length.py::length_reward"
```

| File | Recipe class | Task | Required columns |
|------|---|---|---|
| `vlm_grounding.py` | `VLMGroundingRecipe` | VLM visual grounding with JSON-output bounding boxes, CIoU vs `bbox_gt`, Hungarian matching over `bboxes_gt` with over- and under-prediction penalties. | `prompt`, `bbox_gt`, `bboxes_gt` |

### Writing your own recipe

Drop a new file in `rewards/`. Define your reward functions in the same
file as the class that uses them — everything about the task is in one
place. Subclass `Recipe` and override `rewards()`:

```python
# rewards/my_task.py
from leap_finetune.rewards import Recipe


def format_reward(completions, **kwargs):
    return [1.0 if "<answer>" in (c[0]["content"] if isinstance(c, list) else c) else 0.0
            for c in completions]


def correctness_reward(completions, solution, **kwargs):
    # ...your verification logic...
    return [...]


class MyTaskRecipe(Recipe):
    """One-line summary shown in logs when the recipe is loaded."""

    description = "My task"
    required_columns = ("prompt", "solution")
    system_prompt = "You are a ... Respond with ..."

    def rewards(self):
        return [
            (format_reward,      0.2),
            (correctness_reward, 1.0),
        ]
```

Then reference it from YAML:

```yaml
rewards:
  recipe: "./rewards/my_task.py::MyTaskRecipe"
```

### Extending a shipped recipe

Use `load_recipe` to pull in a sibling recipe class as a parent, then
subclass it with regular Python inheritance. This is the right pattern
when the shipped recipe is *mostly* what you want but you need to add or
replace a reward.

```python
# rewards/my_grounding_with_captions.py
from leap_finetune.rewards import load_recipe

VLMGroundingRecipe = load_recipe(
    "./rewards/vlm_grounding.py::VLMGroundingRecipe"
)


def description_judge_reward(completions, object_descriptions, **kwargs):
    """Score per-object descriptions with an LLM judge."""
    return [...]


class GroundingWithCaptionsRecipe(VLMGroundingRecipe):
    """Base grounding rewards + an LLM-judge for object descriptions."""

    required_columns = VLMGroundingRecipe.required_columns + ("object_descriptions",)

    def rewards(self):
        return [
            *super().rewards(),
            (description_judge_reward, 0.3),
        ]
```

```yaml
rewards:
  recipe: "./rewards/my_grounding_with_captions.py::GroundingWithCaptionsRecipe"
```

That's the whole extension story — plain Python subclassing, no registry,
no decorators, no codegen. If you want to *reweight* the parent's rewards
instead of adding new ones, override `rewards()` and return a different
list of `(callable, float)` tuples. If you want to *remove* a reward,
return a filtered version of `super().rewards()`.

## Notes

- Reward functions are discovered and imported once per training run when
  the config is parsed (driver-side). They are then sent to each Ray Train
  worker as part of the trainer state.
- For dependencies that aren't in `pyproject.toml` (like `math_verify` for
  `accuracy.py`), install them yourself: `uv pip install math_verify`. We
  intentionally don't pull these into the base install.
- Reward functions can be `async def` if they need to call external services
  — TRL runs async rewards concurrently via `asyncio.gather`.
- Keep reward functions deterministic and side-effect-free where possible —
  they're called multiple times per training step.
