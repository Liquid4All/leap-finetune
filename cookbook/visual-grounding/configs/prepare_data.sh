#!/bin/bash
# CPU-only SLURM job that downloads MGrounding-630k from the HF Hub,
# canonicalizes every sample (including the multi-image Object_Tracking
# subset) into our [{"label","bbox"}] JSON schema, and writes a disjoint
# 80/20 SFT-vs-GRPO split with a 10% test holdout. See prepare_data.py
# for the conversion details.

#SBATCH --job-name=grounding-prep-data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/grounding/PREP_OUT_%x.%j
#SBATCH --error=logs/grounding/PREP_ERR_%x.%j

set -euo pipefail

mkdir -p logs/grounding

if [ -f "$HOME/.env" ]; then
    set -a; source "$HOME/.env"; set +a
fi

# Pick a cache location that has enough free space for the raw HF dataset
# (~30 GB after Object_Tracking is extracted). Override HF_HOME if your
# $HOME is small.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/.cache/tmp"
mkdir -p "$TMPDIR"

export PYTHONUNBUFFERED=1

uv run python cookbook/visual-grounding/prepare_data.py \
    --output ./data \
    --cache-dir "$HF_HOME" \
    --grpo-fraction 0.18 \
    --test-fraction 0.10 \
    --seed 42
