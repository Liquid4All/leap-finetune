---
name: lft-checkpoints
description: Inspect, resume, export, and validate leap-finetune checkpoints. Use for output dirs, resume_from_checkpoint, HF exports, manual sharded checkpoints, GGUF export, and checkpoint tests.
---

# LFT Checkpoints

## Use When

- Resuming interrupted training.
- Inspecting `outputs/{project_name}/{run_name}/`.
- Exporting HF or GGUF artifacts.
- Validating MoE expert checkpoint layout.
- Debugging checkpoint callback, model loading, or metadata.

## Do Not Use When

- Launching a new backend job only. Use `lft-launch`.
- Editing dataset or reward contracts. Use `lft-data` or `lft-rewards`.

## Canonical Files

- `README.md`
- `src/leap_finetune/checkpointing/`
- `src/leap_finetune/cli/export_manual_sharded_checkpoint.py`
- `src/leap_finetune/cli/export_gguf.py`
- `tests/test_checkpoint_callback.py`
- `tests/test_hf_export_metadata.py`

## Steps

1. Find the run directory and latest checkpoint under
   `outputs/{project_name}/{run_name}/` or the configured `output_dir`.
2. For resume, set `training_config.resume_from_checkpoint` to `latest` or an
   explicit checkpoint path.
3. Keep `save_only_model: false` when optimizer-state resume is required.
4. For GGUF export, use `uv run leap-export-gguf`.
5. For manual sharded checkpoint export, use
   `uv run leap-export-manual-sharded-checkpoint`.
6. Keep exported models and checkpoint artifacts out of git.

## Expected Output

- The resume path, export command, checkpoint inspection result, or code patch.
- The artifact location and whether it is git-ignored or external storage.
- The validation command for HF, MoE, or GGUF work.
- Any limitation that affects optimizer-state resume.

## Verification

```bash
uv run pytest tests/test_checkpoint_callback.py tests/test_hf_export_metadata.py
```

For MoE exported checkpoints:

```bash
uv run python tests/checkpoint_validation/validate_hf_checkpoint.py /path/to/checkpoint --expected-num-experts <n>
```

## Common Failures

- `latest` resolves under `outputs/{project_name}/`; custom output dirs need
  explicit paths.
- Adapter K-quants require merging the adapter into the base model before GGUF
  export.
- Large checkpoint exports should use persistent shared storage such as
  `/lambdafs`, not `/tmp`.
