#!/bin/bash
# Single-node SLURM launcher for the grounding SFT cookbook.
#
# Assumes 8 GPUs on one node. Run prepare_data.py + prepare_evals.py
# once before this script — see cookbook/visual-grounding/README.md.

#SBATCH --job-name=grounding-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
# Add `#SBATCH --exclude=<node>` lines if your cluster has known-bad GPUs.
#SBATCH --output=logs/grounding/OUT_%x.%j
#SBATCH --error=logs/grounding/ERR_%x.%j

set -euo pipefail

mkdir -p logs/grounding

source .venv/bin/activate

# Cluster-specific CUDA module — set $LEAP_CUDA_MODULE if your cluster
# uses `module load`.
if [ -n "${LEAP_CUDA_MODULE:-}" ] && command -v module >/dev/null 2>&1; then
    module load "$LEAP_CUDA_MODULE" 2>/dev/null || true
fi

# Inherit user-level secrets (WANDB_API_KEY, HF_TOKEN, HF_HOME, etc.)
if [ -f "$HOME/.env" ]; then
    source "$HOME/.env"
fi

# Triton JIT cache: redirect to $HOME if /var/tmp isn't writable on
# your compute nodes (common on shared clusters). Safe to skip otherwise.
export TMPDIR="${TMPDIR:-$HOME/.cache/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

export PYTHONUNBUFFERED=1
# Compute nodes occasionally fail huggingface.co DNS lookups. The model
# weights + processor are already in HF_HOME cache, so force offline
# mode to avoid LFM2.5-VL's custom_generate/generate.py fetch failing.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uv run leap-finetune cookbook/visual-grounding/configs/sft_grounding.yaml
