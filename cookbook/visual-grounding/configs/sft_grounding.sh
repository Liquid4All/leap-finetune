#!/bin/bash
# Visual Grounding SFT launcher — single node, 8 GPUs.

#SBATCH --job-name=grounding-sft
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

if [ -n "${LEAP_CUDA_MODULE:-}" ] && command -v module >/dev/null 2>&1; then
    module load "$LEAP_CUDA_MODULE" 2>/dev/null || true
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

if [ -f "$HOME/.env" ]; then
    source "$HOME/.env"
fi

# See grpo_grounding.sh for why TMPDIR/TRITON_CACHE_DIR is unconditional.
export TMPDIR="$HOME/.cache/tmp"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uv run leap-finetune cookbook/visual-grounding/configs/sft_grounding.yaml
