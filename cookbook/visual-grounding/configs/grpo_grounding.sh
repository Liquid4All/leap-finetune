#!/bin/bash
# Visual Grounding GRPO launcher — single node, 8 GPUs.

#SBATCH --job-name=grounding-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/grounding/OUT_%x.%j
#SBATCH --error=logs/grounding/ERR_%x.%j

set -euo pipefail

mkdir -p logs/grounding

source .venv/bin/activate

# Optional: load your cluster's CUDA module if `module load` is in use.
if [ -n "${LEAP_CUDA_MODULE:-}" ] && command -v module >/dev/null 2>&1; then
    module load "$LEAP_CUDA_MODULE" 2>/dev/null || true
fi
# DeepSpeed JIT-compiles CUDA ops at startup — needs CUDA_HOME to find nvcc.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# Optional secrets file (WANDB_API_KEY, HF_TOKEN, HF_HOME, ...).
if [ -f "$HOME/.env" ]; then
    source "$HOME/.env"
fi

# Some launch environments (tmux/screen/vscode-remote/Claude Code) export a
# host-local TMPDIR that doesn't exist on compute nodes. Child sbatch jobs
# (e.g. sidecar evals) inherit that and die in slurmstepd before any user
# code runs. Force a stable per-user path; change if your cluster prefers
# /scratch/$USER or similar.
export TMPDIR="$HOME/.cache/tmp"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uv run leap-finetune cookbook/visual-grounding/configs/grpo_grounding.yaml
