#!/bin/bash
# Toy slurm launcher for async_eval mode=reserved.
# Allocates 3 GPUs total: 2 for training, 1 carved off at job start for the
# long-running trl vllm-serve subprocess that the helper thread manages.
#SBATCH --job-name=eval_toy_reserved
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-gpu=14
#SBATCH --time=00:45:00
#SBATCH --output=logs/async_eval_toy/OUT_reserved_%j.log
#SBATCH --error=logs/async_eval_toy/ERR_reserved_%j.log

set -euo pipefail

module load cuda12.9/toolkit/12.9.1 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda12.9/toolkit/12.9.1}"

cd /home/rouzbeh/leap-finetune
source .venv/bin/activate
source ~/.env

export TMPDIR="${HOME}/tmp"
export TRITON_CACHE_DIR="${HOME}/.triton_cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" logs/async_eval_toy

uv run leap-finetune tests/fixtures/toy_async_eval_reserved.yaml
