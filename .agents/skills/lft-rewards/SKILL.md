---
name: lft-rewards
description: Build GRPO reward functions, judge rewards, and reward recipes for leap-finetune. Use when editing rewards, reward YAML, RL task bundles, or reward tests.
---

# LFT Rewards

## Use When

- Adding a primitive reward in `rewards/`.
- Adding a task recipe under `rewards/tasks/`.
- Wiring `rewards:` YAML for GRPO or VLM GRPO.
- Adding LLM-as-judge reward config.
- Testing reward resolution, signatures, or recipe behavior.

## Do Not Use When

- Creating benchmark eval metrics. Use `lft-evals`.
- Editing general training config only. Use `lft-training`.

## Canonical Files

- `rewards/README.md`
- `rewards/tasks/README.md`
- `src/leap_finetune/rl/rewards/`
- `src/leap_finetune/rl/judge.py`
- `tests/rl/test_rewards_loader.py`
- `tests/rl/test_judge_llm.py`

## Steps

1. Prefer a small primitive reward when a single function is enough.
2. Use a task recipe when the task needs multiple rewards, default weights,
   required columns, or a recommended system prompt.
3. Keep reward functions deterministic and side-effect-free.
4. Accept `completions` and `**kwargs`; dataset columns are forwarded through
   `kwargs`.
5. Return one float or `None` per completion.
6. For judge rewards, configure `rewards.judge`; use `base_url` for an external
   judge or reserve `grpo_rollout.judge_gpus` for a local vLLM judge.
7. Update `rewards/README.md` or `rewards/tasks/README.md` when adding shipped
   rewards.

## Expected Output

- A primitive reward, task recipe, judge config, or YAML `rewards:` snippet.
- The reward signature and required dataset columns.
- Weight ordering when recipes and primitive rewards are combined.
- The focused reward tests to run.

## Verification

```bash
uv run pytest tests/rl/test_rewards_loader.py tests/rl/test_judge_llm.py
```

For GRPO data/reward integration:

```bash
uv run pytest tests/rl
```

## Common Failures

- `weights` must match the expanded recipe-plus-functions reward list.
- Required dataset columns are documented by recipes but not auto-created.
- Async reward functions are allowed, but should still be deterministic.
- External judge endpoints must speak the TRL vLLM `/generate/` protocol.
