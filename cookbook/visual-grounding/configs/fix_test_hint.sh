#!/bin/bash
# CPU-only SLURM job: rewrite the mgrounding test parquet so every row
# uses the canonical EVAL_FORMAT_HINT (matching RefCOCO/+/g evals).
# Idempotent — re-running is safe.

#SBATCH --job-name=grounding-fix-hint
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/grounding/FIX_HINT_OUT_%j
#SBATCH --error=logs/grounding/FIX_HINT_ERR_%j

set -euo pipefail

mkdir -p logs/grounding

if [ -f "$HOME/.env" ]; then
    set -a; source "$HOME/.env"; set +a
fi

export PYTHONUNBUFFERED=1

uv run python cookbook/visual-grounding/fix_test_hint.py \
    --parquet ./job_datasets/grounding-cookbook/data/grounding_test/test.parquet
