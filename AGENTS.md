# Leap Finetune Agent Guide

## Overview

`leap-finetune` is a local-first model factory for customizing LFMs. Keep the
direct Python/CLI training surface small and reliable while making it easy for
Codex, Claude Code, and similar agents to create data, evals, rewards, launch
training, inspect runs, and iterate on model candidates.

This repo currently prioritizes the standalone OSS workflow. Hosted backends
and proprietary data-generation recipes are future product surfaces, not the
default assumption for repo changes.

## Core Commands

- Install CUDA/default environment: `uv sync`
- Install ROCm environment: `uv sync --no-group cuda --group rocm`
- Run training: `uv run leap-finetune job_configs/sft_example.yaml`
- Run explicit training command: `uv run leap-finetune run job_configs/sft_example.yaml`
- Run standalone eval: `uv run leap-finetune eval job_configs/eval_standalone_example.yaml --output results.json`
- Generate SLURM script: `uv run leap-finetune slurm job_configs/sft_example_with_slurm.yaml`
- Export GGUF: `uv run leap-export-gguf /path/to/checkpoint --output-dir /path/to/gguf`
- Run core non-GPU tests: `uv run pytest tests/config tests/distribution tests/evaluation tests/rl tests/moe`

## Agent Workflow

- Use repo skills in `.agents/skills/` for multi-step Leap workflows.
- Prefer editing YAML configs, reward recipes, eval configs, and dataset
  validation code over adding broad new abstractions.
- Keep generated data, checkpoints, exported models, and run outputs out of
  git.
- For training or eval changes, add or update a focused test when the behavior
  can be checked without a GPU.
- For GPU-only behavior, add a small config, fixture, or documented smoke path
  that a cluster run can execute later.

## Skill Selection

| Task | Skill |
| --- | --- |
| Create or modify a training config or trainer path | `lft-training` |
| Add eval suites, metrics, fixtures, or async eval behavior | `lft-evals` |
| Add GRPO rewards, task recipes, or judge rewards | `lft-rewards` |
| Prepare or validate training/eval data formats | `lft-data` |
| Launch or debug Modal, SLURM, KubeRay, or local backend dispatch | `lft-launch` |
| Resume, inspect, export, or validate checkpoints | `lft-checkpoints` |

## Storage

- Do not use `/tmp` for large materializations such as model downloads, HF caches, tokenized datasets, checkpoint exports, vLLM assets, or package caches.
- Prefer repo-local caches for small/reusable artifacts and persistent shared storage such as `/lambdafs` for large model/checkpoint artifacts.
- `/tmp` is acceptable only when the tool requires node-local scratch space or short socket paths, such as Ray runtime/session directories.

## Useful Files

- `README.md`: user-facing setup, CLI, backend, dataset, GRPO, eval, and export guide.
- `job_configs/`: starter YAML configs for SFT, DPO, GRPO, VLM, MoE, Modal, SLURM, and evals.
- `rewards/README.md`: reward primitives, task recipes, judge rewards, and custom reward contract.
- `src/leap_finetune/evaluation/README.md`: eval config and metric contract.
- `tests/README.md`: test buckets and local/GPU/SLURM check commands.
