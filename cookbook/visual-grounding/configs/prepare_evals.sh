#!/bin/bash
# CPU-only SLURM job for the grounding cookbook eval data prep.
#
# Runs prepare_evals.py on a compute node: downloads RefCOCO + RefCOCO+
# + RefCOCOg val splits from HuggingFace and writes one jsonl per
# benchmark. Lightweight (~500 samples each, ~1 GB total disk).

#SBATCH --job-name=grounding-prep-evals
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/grounding/EVAL_PREP_OUT_%j
#SBATCH --error=logs/grounding/EVAL_PREP_ERR_%j

set -euo pipefail

mkdir -p logs/grounding

if [ -f "$HOME/.env" ]; then
    set -a; source "$HOME/.env"; set +a
fi

export HF_HOME=./job_datasets/grounding-cookbook/cache
export PYTHONUNBUFFERED=1

uv run python cookbook/visual-grounding/prepare_evals.py \
    --output ./job_datasets/grounding-cookbook/data/grounding_evals \
    --limit 5000  # full splits (3811 + 3805 + 2573) — tight ±1% noise floor
