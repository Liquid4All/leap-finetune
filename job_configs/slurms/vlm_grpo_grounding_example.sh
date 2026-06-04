#!/bin/bash
# 2-node SLURM launcher for job_configs/vlm_grpo_grounding_example.yaml.
#
# Starts a Ray cluster spanning both SLURM nodes, waits for all workers
# to register, then runs leap-finetune. trainer.py auto-detects the
# existing cluster via the RAY_ADDRESS env var and skips its local init.
#
# Adjust SBATCH directives to your cluster (partition, account, GPU
# count per node). The example assumes 8 GPUs/node × 2 nodes = 16 GPUs.

#SBATCH --job-name=vlm-grpo-grounding-example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/grpo/OUT_%x.%j
#SBATCH --error=logs/grpo/ERR_%x.%j
#SBATCH --cpus-per-gpu=14

set -euo pipefail

source .venv/bin/activate

# Uncomment if your cluster requires loading a CUDA module:
# module load cuda/12.9

export PYTHONUNBUFFERED=1

# Multi-node Ray helpers shipped with leap-finetune — start a cluster
# on the SLURM allocation, wait for it to be ready, export
# RAY_ADDRESS, then let trainer.py connect to it.
# shellcheck source=utils/slurm_ray.sh
source job_configs/slurms/utils/slurm_ray.sh

ray_slurm_init "${SLURM_NNODES}" "${SLURM_GPUS_PER_NODE}"
ray_slurm_start_cluster_bg
ray_slurm_wait_ready "${SLURM_NNODES}" "${TOTAL_GPUS}" 600 5

echo "Ray cluster up: ${TOTAL_GPUS} GPUs across ${SLURM_NNODES} nodes (RAY_ADDRESS=${RAY_ADDRESS})"

export RAY_ADDRESS
uv run leap-finetune job_configs/vlm_grpo_grounding_example.yaml

ray_slurm_stop_cluster
