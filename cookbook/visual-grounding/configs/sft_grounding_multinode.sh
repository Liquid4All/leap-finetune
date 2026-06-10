#!/bin/bash
# Visual Grounding SFT launcher — 2 nodes, 16 GPUs.
# Brings up a Ray cluster, exports RAY_ADDRESS, runs leap-finetune.

#SBATCH --job-name=grounding-sft-2n
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=14
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

# See grpo_grounding.sh for why this is unconditional.
export TMPDIR="$HOME/.cache/tmp"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Multi-node Ray helpers shipped with leap-finetune.
# shellcheck source=../../../job_configs/slurms/utils/slurm_ray.sh
source job_configs/slurms/utils/slurm_ray.sh

ray_slurm_init "${SLURM_NNODES}" "${SLURM_GPUS_PER_NODE}"
ray_slurm_start_cluster_bg
ray_slurm_wait_ready "${SLURM_NNODES}" "${TOTAL_GPUS}" 600 5

echo "Ray cluster up: ${TOTAL_GPUS} GPUs across ${SLURM_NNODES} nodes (RAY_ADDRESS=${RAY_ADDRESS})"

export RAY_ADDRESS
uv run leap-finetune cookbook/visual-grounding/configs/sft_grounding.yaml

ray_slurm_stop_cluster
