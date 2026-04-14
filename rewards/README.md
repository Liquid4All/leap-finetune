# Rewards

Two layers:

1. **Primitives** at the root — small, single-file reward functions you
   compose from YAML (`accuracy.py`, `length.py`).
2. **Task bundles** under [`tasks/`](tasks/README.md) — complete
   recipes for concrete datasets (VLM grounding, GSM8K, IFEval, MCQA).
   Each task lives in its own folder with a single `recipe.py`.

For the recipe index and copy-paste YAML snippets, go to
[`tasks/README.md`](tasks/README.md).

## Referencing rewards from YAML

### As a recipe (task bundle)

```yaml
rewards:
  recipe: "./rewards/tasks/vlm_grounding/recipe.py::VLMGroundingIoURecipe"
```

### As a list of primitives

```yaml
rewards:
  funcs:
    - "./rewards/accuracy.py::accuracy_reward"
    - "./rewards/length.py::length_reward"
  weights: [1.0, 0.2]
```

Paths are resolved relative to the current working directory first,
then to the directory containing the YAML. Absolute paths work too.

### Stacking a recipe with extra primitives

```yaml
rewards:
  recipe: "./rewards/tasks/vlm_grounding/recipe.py::VLMGroundingIoURecipe"
  funcs:
    - "./rewards/length.py::length_reward"
  weights: [0.1, 1.0, 0.05]   # recipe weights + the stacked func weight
```

## Shipped primitives

| File | Function | What it does | Required columns |
|------|----------|--------------|------------------|
| `accuracy.py` | `accuracy_reward` | Math accuracy via `math_verify` (re-export of `trl.rewards.accuracy_reward`). | `solution` (str) |
| `length.py` | `length_reward` | Length-based shaping reward, scaled to `[0, 1]`. | none |

## Shipped task bundles

Full list in [`tasks/README.md`](tasks/README.md).

| Task | Recipe | Reward shape |
|------|--------|--------------|
| VLM visual grounding | `tasks/vlm_grounding/recipe.py::VLMGroundingIoURecipe` | strict JSON format (0.1) + F1 of matched IoUs (1.0) |
| VLM visual grounding (CIoU) | `tasks/vlm_grounding/recipe.py::VLMGroundingCIoURecipe` | strict JSON format (0.1) + F1 of matched CIoUs (1.0) |
| GSM8K | `tasks/gsm8k/recipe.py::GSM8KRecipe` | numeric exact match via `#### N` (1.0) |
| MCQA | `tasks/mcqa/recipe.py::MCQARecipe` | letter match A..J (1.0) |
| IFEval | `tasks/ifeval/recipe.py::IFEvalRecipe` | fraction of constraints satisfied (1.0) |

## Reward function signature

```python
def reward_fn(completions, **kwargs) -> list[float | None]:
    ...
```

- **`completions`** — one entry per generation. Conversational prompts
  wrap each entry as `[{"role": "assistant", "content": "<text>"}]`;
  string prompts pass through the raw string. Extract defensively:
  `c[0]["content"] if isinstance(c, list) else c`.
- **`**kwargs`** — every other column in the dataset row is forwarded
  as a keyword with the same name. TRL also forwards `prompts`,
  `completion_ids`, `trainer_state`, and (when `rl_env` is used)
  `env_reward`. Use `**kwargs` so unused fields are ignored.
- **Return** — a list of floats, one per completion. Returning `None`
  for a sample marks it "not applicable" and drops it from advantage
  aggregation.

## Writing a custom reward

### As a primitive

Drop a new `.py` file at the root of `rewards/` and reference it by
path:

```python
# rewards/my_primitive.py
def my_primitive_reward(completions, **kwargs):
    contents = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return [1.0 if len(t) >= 50 else 0.0 for t in contents]
```

```yaml
rewards:
  funcs:
    - "./rewards/my_primitive.py::my_primitive_reward"
```

### As a task bundle

Create a folder under `tasks/` with `__init__.py` and `recipe.py`:

```python
# rewards/tasks/my_task/recipe.py
from leap_finetune.rewards import Recipe


def my_correctness_reward(completions, solution, **kwargs):
    ...


class MyTaskRecipe(Recipe):
    description = "My task — what the reward measures"
    required_columns = ("prompt", "solution")
    system_prompt = "..."

    def rewards(self):
        return [(my_correctness_reward, 1.0)]
```

```python
# rewards/tasks/my_task/__init__.py
from .recipe import MyTaskRecipe, my_correctness_reward

__all__ = ["MyTaskRecipe", "my_correctness_reward"]
```

```yaml
rewards:
  recipe: "./rewards/tasks/my_task/recipe.py::MyTaskRecipe"
```

Then add a row to [`tasks/README.md`](tasks/README.md).

### Extending a shipped recipe

Load a sibling recipe as a parent and subclass it:

```python
# rewards/tasks/my_grounding_plus_captions/recipe.py
from leap_finetune.rewards import load_recipe

VLMGroundingIoURecipe = load_recipe(
    "./rewards/tasks/vlm_grounding/recipe.py::VLMGroundingIoURecipe"
)


def description_judge_reward(completions, object_descriptions, **kwargs):
    ...


class GroundingWithCaptionsRecipe(VLMGroundingIoURecipe):
    required_columns = VLMGroundingIoURecipe.required_columns + ("object_descriptions",)

    def rewards(self):
        return [
            *super().rewards(),
            (description_judge_reward, 0.3),
        ]
```

To reweight the parent without adding new rewards, override `rewards()`
and return different `(callable, float)` tuples. To remove a reward,
return a filtered version of `super().rewards()`.

## Notes

- Reward functions are imported once per training run when the config
  is parsed, then shipped to each worker as part of the trainer state.
- Dependencies outside `pyproject.toml` (like `math_verify` for
  `accuracy.py`) need to be installed separately.
- Reward functions can be `async def` — TRL runs async rewards
  concurrently via `asyncio.gather`.
- Keep reward functions deterministic and side-effect-free; they run
  multiple times per training step.
