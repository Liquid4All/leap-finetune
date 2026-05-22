#!/bin/bash
# Toy slurm launcher for async_eval mode=sidecar.
# Training takes 2 GPUs on one node. The sidecar callback will sbatch ITS
# OWN 1-GPU eval job per eval_step, separate from this allocation.
#SBATCH --job-name=eval_toy_sidecar
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=14
#SBATCH --time=00:45:00
#SBATCH --output=logs/async_eval_toy/OUT_sidecar_%j.log
#SBATCH --error=logs/async_eval_toy/ERR_sidecar_%j.log

set -euo pipefail

module load cuda12.9/toolkit/12.9.1 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda12.9/toolkit/12.9.1}"

cd /home/rouzbeh/leap-finetune
source .venv/bin/activate
source ~/.env

# Make sure the sidecar runner picks up the same caches
export TMPDIR="${HOME}/tmp"
export TRITON_CACHE_DIR="${HOME}/.triton_cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" logs/async_eval_toy

uv run leap-finetune tests/fixtures/toy_async_eval_sidecar.yaml
