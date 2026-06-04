#!/bin/bash
# Single-node SLURM launcher for job_configs/grpo_example.yaml.
#
# Adjust SBATCH directives to your cluster (partition, account, GPU count).
# Runs the text GRPO example on one node with vLLM colocated on the
# training GPUs.

#SBATCH --job-name=grpo-example
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/grpo/OUT_%x.%j
#SBATCH --error=logs/grpo/ERR_%x.%j
#SBATCH --cpus-per-gpu=14

set -euo pipefail

# Activate your virtualenv (leap-finetune uses uv, so `.venv` is typical).
source .venv/bin/activate

# Uncomment if your cluster requires loading a CUDA module:
# module load cuda/12.9

uv run leap-finetune job_configs/grpo_example.yaml
