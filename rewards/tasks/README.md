# Task recipes

One folder per task. Each folder contains a single `recipe.py` with
the reward functions and a `Recipe` class that wires them up with
default weights. Reference a recipe from YAML with:

```yaml
rewards:
  recipe: "./rewards/tasks/<task>/recipe.py::<RecipeName>"
```

## Shipped recipes

### VLM visual grounding

Bounding-box grounding where the model outputs a bare JSON array of
`{"label", "bbox"}` dicts in normalized `[0, 1]` coordinates.

| Recipe | Reward | Default weights |
|---|---|---|
| `VLMGroundingIoURecipe` | Strict JSON format check + F1 of Hungarian-matched IoUs. | `strict_format_reward: 0.1`, `iou_f1_reward: 1.0` |
| `VLMGroundingCIoURecipe` | Same F1 structure, but each matched pair uses Complete-IoU (IoU − center-distance − aspect-ratio). The Hungarian matcher also runs on CIoU. | `strict_format_reward: 0.1`, `ciou_f1_reward: 1.0` |

**Required columns:** `prompt` (VLM messages list with image +
instruction), `solution` (JSON array of `{label, bbox}` dicts).

**Example:**

```yaml
project_name: "vlm_grpo_grounding"
model_name: "LFM2-VL-1.6B"
training_type: "vlm_grpo"

dataset:
  path: "your-org/your-grounding-parquet"
  type: "vlm_grpo"
  image_root: "/path/to/your/images"

rewards:
  recipe: "./rewards/tasks/vlm_grounding/recipe.py::VLMGroundingIoURecipe"

training_config:
  extends: "DEFAULT_VLM_GRPO"
```

Pick `VLMGroundingCIoURecipe` when center alignment and aspect ratio
matter on top of raw overlap (e.g. single-object tasks where hitting
the wrong visually similar object at the same overlap should be
penalized). Plain IoU-F1 is typically enough for multi-object scenes.

---

### GSM8K — math word problems

Exact-match reward on the final numeric answer, extracted via the
`#### N` marker.

| Recipe | Reward | Default weights |
|---|---|---|
| `GSM8KRecipe` | Extract the final number via `#### N` (with last-number fallback) and compare to the gold. 1.0 / 0.0 / `None`. | `gsm8k_reward: 1.0` |

**Required columns:** `prompt` (question), `solution` (bare numeric
string like `"72"` or a full CoT ending with `"... #### 72"`).

**Example:**

```yaml
project_name: "grpo_gsm8k"
model_name: "LFM2-1.2B"
training_type: "grpo"

dataset:
  path: "/path/to/gsm8k/train.parquet"
  type: "grpo"

rewards:
  recipe: "./rewards/tasks/gsm8k/recipe.py::GSM8KRecipe"

training_config:
  extends: "DEFAULT_GRPO"
```

---

### MCQA — multiple-choice question answering

Letter-match reward on the extracted answer choice. Supports
`Answer: X` / `\boxed{X}` / trailing-letter patterns; the last match
in the completion wins.

| Recipe | Reward | Default weights |
|---|---|---|
| `MCQARecipe` | Extract a letter A..J from the completion tail and compare to the gold. 1.0 / 0.0 / `None`. | `mcqa_reward: 1.0` |

**Required columns:** `prompt` (question + labeled options),
`solution` (bare letter or a sentence the letter can be parsed from).

**Example:**

```yaml
project_name: "grpo_mcqa"
model_name: "LFM2-1.2B"
training_type: "grpo"

dataset:
  path: "/path/to/mcqa/train.parquet"
  type: "grpo"

rewards:
  recipe: "./rewards/tasks/mcqa/recipe.py::MCQARecipe"

training_config:
  extends: "DEFAULT_GRPO"
```

---

### IFEval — instruction-following constraints

Dense reward in `[0, 1]`: the fraction of supported constraints the
completion satisfies.

| Recipe | Reward | Default weights |
|---|---|---|
| `IFEvalRecipe` | Parse a JSON constraint spec from `solution`, check each supported constraint, return the fraction that pass. | `ifeval_reward: 1.0` |

**Supported constraint types:** `punctuation:no_comma`,
`length_constraints:number_words`,
`detectable_format:number_highlighted_sections`,
`count:keywords_multiple`, `counting:letter_count_in_word`. Unknown
IDs are silently skipped; samples with zero supported constraints
return `None` so GRPO drops them from advantage computation.

**Required columns:** `prompt` (instruction), `solution` (JSON
constraint spec — see the recipe docstring for the schema).

**Example:**

```yaml
project_name: "grpo_ifeval"
model_name: "LFM2-1.2B"
training_type: "grpo"

dataset:
  path: "/path/to/ifeval/train.parquet"
  type: "grpo"

rewards:
  recipe: "./rewards/tasks/ifeval/recipe.py::IFEvalRecipe"

training_config:
  extends: "DEFAULT_GRPO"
```

---

## Composing with primitives

Recipes can be stacked with primitives from `rewards/`:

```yaml
rewards:
  recipe: "./rewards/tasks/gsm8k/recipe.py::GSM8KRecipe"
  funcs:
    - "./rewards/length.py::length_reward"
  weights: [1.0, 0.2]
```

See [`../README.md`](../README.md) for the full list of primitives
and the reward-function API contract.

## Adding a new task

1. Create `rewards/tasks/<your_task>/` with an `__init__.py` and a
   `recipe.py`.
2. Write the reward functions and a `Recipe` subclass in `recipe.py`.
   Keep parsing/geometry helpers private (leading underscore).
3. Re-export from `__init__.py`:

   ```python
   from .recipe import YourTaskRecipe, your_reward_fn

   __all__ = ["YourTaskRecipe", "your_reward_fn"]
   ```
4. Add a row to this README.
5. Reference the recipe from YAML:
   `./rewards/tasks/<your_task>/recipe.py::YourTaskRecipe`.
