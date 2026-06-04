#!/bin/bash
# CPU-only SLURM job that downloads RefCOCO + RefCOCO+ + RefCOCOg val
# splits from the HF Hub and writes one jsonl per benchmark.
# Lightweight (~500 samples each at default --limit, ~1 GB total disk).

#SBATCH --job-name=grounding-prep-evals
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

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

# Full val splits (3811 + 3805 + 2573) keep the ±1 IoU-point noise floor.
uv run python cookbook/visual-grounding/prepare_evals.py \
    --output ./data/grounding_evals \
    --limit 5000
