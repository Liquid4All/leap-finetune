#!/bin/bash
#SBATCH --job-name=leap-grpo-configs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --account=liquidai
#SBATCH --output=logs/grpo_configs_%j.out

# Config / reward loader / OpenEnv adapter unit tests — no GPU needed.
# You can also run these directly on any node where the venv has torch
# installed correctly. This sbatch wrapper is here for convenience so the
# whole test suite can be kicked off with a single `sbatch` call on the
# cluster.

set -e
cd "$(dirname "$0")/.."
mkdir -p logs
module load cuda12.9/toolkit/12.9.1
source .venv/bin/activate

uv run pytest --configs -v \
    tests/test_rewards_loader.py \
    tests/test_rl_env_adapter.py \
    tests/test_grpo_config.py

uv run pytest --data -v tests/test_grpo_data.py

echo "CONFIG / DATA SUITE DONE"
