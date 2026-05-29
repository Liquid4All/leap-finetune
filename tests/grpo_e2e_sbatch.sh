#!/bin/bash
#SBATCH --job-name=leap-grpo-e2e
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=00:30:00
#SBATCH --account=liquidai
#SBATCH --output=logs/grpo_e2e_%j.out

# End-to-end GRPO smoke-test launcher.
#
# What it runs:
#   1. tests/test_grpo_e2e.py        — text GRPO, 1 H100, colocate vLLM
#   2. tests/test_vlm_grpo_e2e.py    — VLM GRPO, 1 H100, colocate vLLM
#
# Submit with:
#   sbatch tests/grpo_e2e_sbatch.sh
#
# Tail the logs with:
#   tail -f logs/grpo_e2e_<jobid>.out
#
# Adjust --gpus-per-task if your dev partition requires more, but the tests
# are intentionally tiny and will fit on a single GPU.

set -e

cd "$(dirname "$0")/.."        # cd into repo root
mkdir -p logs

# Load CUDA toolkit so torch can find libcudnn.so.9 etc. Failing hard if the
# module isn't available — a missing CUDA toolkit will lead to confusing
# torch import errors later.
module load cuda12.9/toolkit/12.9.1

# Activate the project venv
source .venv/bin/activate

# Sanity: verify GPU is visible
python - <<'PY'
import torch
print(f"torch {torch.__version__}, cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
assert torch.cuda.is_available(), "GPU not visible inside SLURM job"
PY

echo
echo "=== Text GRPO E2E ==="
uv run pytest --dense tests/test_grpo_e2e.py -v -s

echo
echo "=== VLM GRPO E2E ==="
uv run pytest --vlm tests/test_vlm_grpo_e2e.py -v -s

echo
echo "GRPO E2E SUITE DONE"
