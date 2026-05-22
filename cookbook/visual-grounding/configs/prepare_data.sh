#!/bin/bash
# CPU-only SLURM job for the grounding cookbook data prep.
#
# Runs prepare_data.py on a compute node:
#   1. snapshot_download Michael4933/MGrounding-630k (~140 GB)
#   2. Extract zips (7z for multi-part Group_Grounding; skip Object_Tracking)
#   3. Parse manifest, convert to leap-finetune messages parquet
#   4. Deterministic 3-way split: SFT (65%) / GRPO (25%) / test (10%)
#
# Compute on the head node would hammer shared CPU + I/O for hours;
# always sbatch this. Use a CPU-only allocation — no GPU needed.

#SBATCH --job-name=grounding-prep-data
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/grounding/PREP_OUT_%j
#SBATCH --error=logs/grounding/PREP_ERR_%j

set -euo pipefail

mkdir -p logs/grounding

# Multi-volume zip extraction needs 7-Zip. If your compute nodes don't
# have it system-wide, drop a `7zz` binary somewhere on $PATH first.

# Inherit user-level secrets (HF_TOKEN, etc.)
if [ -f "$HOME/.env" ]; then
    set -a; source "$HOME/.env"; set +a
fi

# Pin HF cache to the same root so the 140 GB download lands somewhere
# with enough free space (typical $HOME is too small).
export HF_HOME=./job_datasets/grounding-cookbook/cache
export TMPDIR="${TMPDIR:-$HOME/.cache/tmp}"
mkdir -p "$TMPDIR"

export PYTHONUNBUFFERED=1

uv run python cookbook/visual-grounding/prepare_data.py \
    --output ./job_datasets/grounding-cookbook/data \
    --cache-dir ./job_datasets/grounding-cookbook/cache \
    --grpo-fraction 0.25 \
    --test-fraction 0.10 \
    --seed 42
