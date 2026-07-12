---
name: lft-evals
description: Create, modify, and run leap-finetune eval suites and metrics. Use for standalone eval configs, training evals, async eval, benchmark data, and evaluation tests.
---

# LFT Evals

## Use When

- Adding an `evals:` block to a training config.
- Creating standalone eval configs.
- Adding or changing benchmark metrics.
- Debugging sync, sidecar, or reserved async eval behavior.
- Writing eval fixtures or tests.

## Do Not Use When

- Building reward functions for GRPO. Use `lft-rewards`.
- Validating training datasets only. Use `lft-data`.

## Canonical Files

- `src/leap_finetune/evaluation/README.md`
- `job_configs/eval_standalone_example.yaml`
- `job_configs/sft_with_async_eval_example.yaml`
- `src/leap_finetune/evaluation/`
- `tests/evaluation/`
- `tests/e2e/fixtures/`

## Steps

1. Decide whether the eval runs standalone or during training.
2. Use `evals.benchmarks` with explicit `name`, `path`, and `metric`.
3. Keep benchmark data in the same messages format used by training data.
4. For text evals, rely on default `modality: text`; set `modality: vlm` only
   for standalone VLM evals.
5. For async evals, choose `sync`, `sidecar`, or `reserved` based on available
   GPUs and queue behavior.
6. After standalone or training eval, inspect `uv run leap-finetune runs report`
   and `uv run leap-finetune runs show <run_id>` for recorded metrics/log refs.
7. New eval modes must write compact eval status/results through the shared
   state helpers in addition to any W&B/Trackio logging.
8. When adding a metric, register it in the metric dispatch and the relevant
   text or VLM config factory.

## Expected Output

- An `evals:` block, standalone eval config, metric change, or fixture update.
- The exact standalone eval command when applicable.
- The run ID and metrics location for standalone evals.
- State entries for eval submitted/running/completed/failed status where
  relevant.
- The smallest relevant eval test command.
- Any async eval operational notes, especially for SLURM sidecar behavior.

## Verification

```bash
uv run leap-finetune job_configs/eval_standalone_example.yaml --output results.json
uv run pytest tests/evaluation
```

If async behavior changes, also run the relevant fixture or SLURM dry run.

## Common Failures

- Standalone eval configs should not include `dataset`, `training_type`,
  `training_config`, or `async_eval`.
- Generation metrics need a ground-truth assistant turn.
- `logprob_zero_shot` needs `options` and `answer_id`.
- Sidecar eval depends on SLURM availability and should fail without stopping
  the training job.
