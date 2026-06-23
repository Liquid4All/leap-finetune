---
name: lft-training
description: Train LFMs with leap-finetune SFT, DPO, GRPO, VLM, or MoE configs. Use when creating, modifying, validating, or launching training jobs.
---

# LFT Training

## Use When

- Creating or modifying a training YAML in `job_configs/`.
- Choosing between SFT, DPO, GRPO, VLM, MoE, LoRA, or full fine-tuning.
- Launching local training through Ray Train.
- Updating training config parsing, defaults, or trainer code.

## Do Not Use When

- Running standalone benchmark suites without training. Use `lft-evals`.
- Writing reward functions only. Use `lft-rewards`.
- Submitting to Modal, SLURM, or KubeRay only. Use `lft-launch`.

## Canonical Files

- `README.md`
- `job_configs/`
- `src/leap_finetune/config/job_config.py`
- `src/leap_finetune/training/default_configs/`
- `src/leap_finetune/training/`
- `tests/config/`
- `tests/e2e/`

## Steps

1. Identify the training type: `sft`, `dpo`, `grpo`, `vlm_sft`, `vlm_dpo`,
   `vlm_grpo`, `moe_sft`, `moe_dpo`, or expert-parallel MoE.
2. Start from the closest config in `job_configs/` and make the smallest
   config change that expresses the experiment.
3. Let `training_type` select defaults. Put only user overrides in
   `training_config`; enable LoRA with `peft_config.use_peft: true` and add
   only PEFT fields that differ from the default for that training type.
4. Keep config validation in Pydantic models; parser/materialization code
   should resolve paths/defaults and avoid duplicate checks.
5. Keep dataset paths, output paths, and large artifacts outside git.
6. For local training, run `uv run leap-finetune <config>`.
7. After launch, use `uv run leap-finetune runs report` to find progress,
   latest log/eval/checkpoint, and log refs.
8. If adding a new Trainer path or callback, make sure it uses
   `LFTStateCallback` or the shared state helpers.
9. Add config tests only for materialized behavior or launch payloads, not for
   Pydantic validator/default constant assertions.

## Expected Output

- The changed training config, trainer code, or config defaults.
- The exact launch command for local or remote execution.
- The validation or test command that should be run before a real GPU job.
- Notes on output/checkpoint paths when they affect follow-up work.
- The run ID when a job was launched or submitted.
- State updates for progress, evals, checkpoints, and log refs when the change
  affects logging.

## Verification

Run the narrowest relevant check:

```bash
uv run pytest tests/config tests/moe
```

For broad config/backend edits:

```bash
uv run pytest tests/config tests/distribution
```

GPU smoke tests live under `tests/e2e/` and should be run on a suitable node.

## Common Failures

- Local training without CUDA exits early. Add `modal:`, `slurm:`, or
  `kuberay:` for remote execution.
- `flash-attn` ABI errors usually require rebuilding the environment.
- `save_only_model: true` prevents optimizer-state resume.
- Relative paths resolve from the config location first; use absolute paths for
  shared cluster storage.
