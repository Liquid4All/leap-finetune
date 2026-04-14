#!/bin/bash
# Multi-node Ray-on-SLURM helpers for leap-finetune.
#
# leap-finetune's trainer.py auto-detects multi-node mode when RAY_ADDRESS
# is set and connects to the existing cluster instead of doing a local init.

set -euo pipefail

ray_slurm_init() {
  local n_nodes="${1:?n_nodes required}"
  local gpus_per_node="${2:?gpus_per_node required}"

  # Discover head node + its routable IP within this allocation.
  # shellcheck disable=SC2207
  NODES_ARRAY=($(scontrol show hostnames "$SLURM_NODELIST"))
  HEAD_NODE="${NODES_ARRAY[0]}"
  HEAD_IP="$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "hostname -I | cut -d' ' -f1")"

  RAY_PORT="${RAY_PORT:-6379}"
  RAY_ADDRESS="${RAY_ADDRESS:-$HEAD_IP:$RAY_PORT}"
  RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${SLURM_JOB_ID}}"
  RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"

  TOTAL_GPUS=$((n_nodes * gpus_per_node))

  # Prefer SLURM CPU accounting; fall back to a sensible default.
  CPUS_ON_NODE="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-}}"
  if [[ -z "${CPUS_ON_NODE}" ]]; then
    if [[ -n "${SLURM_CPUS_PER_GPU:-}" ]]; then
      CPUS_ON_NODE="$(( SLURM_CPUS_PER_GPU * gpus_per_node ))"
    else
      CPUS_ON_NODE="$(( 14 * gpus_per_node ))"
    fi
  fi

  export NODES_ARRAY HEAD_NODE HEAD_IP
  export RAY_PORT RAY_ADDRESS RAY_TMPDIR RAY_DASHBOARD_PORT
  export TOTAL_GPUS CPUS_ON_NODE
}

# Tear down any stale Ray daemons + temp dirs on every node in the allocation.
# Safe to call before starting a fresh cluster.
ray_slurm_cleanup_all_nodes() {
  RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${SLURM_JOB_ID}}"
  export RAY_TMPDIR

  srun --overlap --nodes="${SLURM_NNODES:-1}" --ntasks-per-node=1 bash -lc "
    uv run --no-sync ray stop >/dev/null 2>&1 || true
    rm -rf /tmp/ray/session_* 2>/dev/null || true
    rm -rf ${RAY_TMPDIR} 2>/dev/null || true
    mkdir -p ${RAY_TMPDIR}
    chmod 700 ${RAY_TMPDIR}
  " >/dev/null 2>&1 || true
}

# Run on every node simultaneously.  The node with SLURM_NODEID=0 becomes the
# Ray head; others join as workers.  --block keeps each ray process alive
# under srun, which is how Ray is supposed to be run on SLURM.
ray_slurm_start_blocking() {
  if [[ "${SLURM_NODEID}" -eq 0 ]]; then
    echo "[ray-head] node=$(hostname) ip=${HEAD_IP} port=${RAY_PORT}"
    uv run --no-sync ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
      --temp-dir "${RAY_TMPDIR}" \
      --dashboard-port="${RAY_DASHBOARD_PORT}" \
      --num-cpus "${CPUS_ON_NODE}" \
      --num-gpus "${SLURM_GPUS_PER_NODE}" \
      --block
  else
    echo "[ray-worker] node=$(hostname) joining ${RAY_ADDRESS}"
    uv run --no-sync ray start --address "${RAY_ADDRESS}" \
      --temp-dir "${RAY_TMPDIR}" \
      --num-cpus "${CPUS_ON_NODE}" \
      --num-gpus "${SLURM_GPUS_PER_NODE}" \
      --block
  fi
}

# Launch the cluster as a backgrounded srun spanning every node, then return
# control so the caller can run the trainer.  Cleans up first to avoid stale
# daemons from a previous job.
ray_slurm_start_cluster_bg() {
  ray_slurm_cleanup_all_nodes

  srun --overlap \
    --nodes="${SLURM_NNODES:-1}" \
    --ntasks-per-node=1 \
    --gpus-per-task="${SLURM_GPUS_PER_NODE}" \
    bash -lc "set -euo pipefail; $(declare -f ray_slurm_start_blocking); ray_slurm_start_blocking" &

  RAY_CLUSTER_PID="$!"
  export RAY_CLUSTER_PID
}

# Stop the backgrounded srun and let `ray stop` propagate via SIGTERM.
ray_slurm_stop_cluster() {
  set +e
  if [[ -n "${RAY_CLUSTER_PID:-}" ]]; then
    kill "${RAY_CLUSTER_PID}" 2>/dev/null || true
    wait "${RAY_CLUSTER_PID}" 2>/dev/null || true
  fi
  set -e
}

# Block until the cluster reports the expected node + GPU counts.
ray_slurm_wait_ready() {
  local expected_nodes="${1:?expected_nodes required}"
  local expected_gpus="${2:?expected_gpus required}"
  local timeout_s="${3:-300}"
  local interval_s="${4:-5}"

  uv run --no-sync python scripts/wait_for_ray.py \
    --address "${RAY_ADDRESS}" \
    --expected-nodes "${expected_nodes}" \
    --expected-gpus "${expected_gpus}" \
    --timeout-s "${timeout_s}" \
    --interval-s "${interval_s}"
}
