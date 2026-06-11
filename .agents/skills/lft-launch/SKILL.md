---
name: lft-launch
description: Launch leap-finetune jobs locally, on SLURM, Modal, or KubeRay. Use when editing backend configs, remote submission, resource planning, logs, or cluster smoke paths.
---

# LFT Launch

## Use When

- Adding or debugging `modal:`, `slurm:`, or `kuberay:` config blocks.
- Generating SLURM scripts.
- Checking remote-submit paths that should avoid heavy GPU imports.
- Updating resource planning or backend tests.
- Explaining how to monitor detached jobs.

## Do Not Use When

- Editing trainer internals without backend dispatch. Use `lft-training`.
- Inspecting completed checkpoint artifacts. Use `lft-checkpoints`.

## Canonical Files

- `README.md`
- `src/leap_finetune/cli/main.py`
- `src/leap_finetune/distribution/backends/`
- `job_configs/sft_example_modal.yaml`
- `job_configs/sft_example_with_slurm.yaml`
- `tests/distribution/`

## Steps

1. Identify backend: local Ray, SLURM, Modal, or KubeRay.
2. Keep remote submission paths light; they should not import torch, Ray,
   PEFT, or datasets unless launching local training.
3. For SLURM, use `uv run leap-finetune slurm <config>` to generate scripts
   without submitting.
4. For Modal, check `detach`, secrets, output volume, GPU string, and printed
   `modal app logs` / `modal app stop` commands.
5. For KubeRay, check namespace, image, worker replicas, GPUs per worker, and
   output PVC.
6. Add backend tests for generated configs, resource planning, or command
   output when behavior changes.

## Expected Output

- The backend config block or backend code change.
- The exact launch, dry-run, monitor, or stop command.
- Whether the job is attached or detached.
- The focused backend test or SLURM generation check.

## Verification

```bash
uv run pytest tests/distribution tests/test_ray_cluster_support.py
uv run leap-finetune slurm job_configs/sft_example_with_slurm.yaml --output-dir job_configs/slurms
```

Do not submit real cluster jobs unless the user asked for it.

## Common Failures

- Local training requires CUDA; use a remote backend from non-GPU laptops.
- Modal detached runs need the printed app ID for logs and stop commands.
- SLURM scripts should write to ignored output locations.
- KubeRay needs a configured Kubernetes context and an image with this repo.
