#!/bin/bash
# Single-node SLURM launcher for the grounding GRPO cookbook (Phase 2).
# Resumes from the Phase 1 SFT checkpoint — update model_name in the YAML
# to point at it before running this script.

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

# Cluster-specific CUDA module — set $LEAP_CUDA_MODULE if needed.
if [ -n "${LEAP_CUDA_MODULE:-}" ] && command -v module >/dev/null 2>&1; then
    module load "$LEAP_CUDA_MODULE" 2>/dev/null || true
fi

if [ -f "$HOME/.env" ]; then
    source "$HOME/.env"
fi

export TMPDIR="${TMPDIR:-$HOME/.cache/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

export PYTHONUNBUFFERED=1

uv run leap-finetune cookbook/visual-grounding/configs/grpo_grounding.yaml
