# Tests

Keep the suite intentionally small. New tests should land in one of these
buckets:

- `config/` — config parsing and generated launch config.
- `distribution/` — launch/resource planning and distributed backend utilities.
- `evaluation/` — benchmark, metric, backend, and async eval contracts.
- `e2e/` — training smoke tests, dense data-path invariants, fixtures, and SLURM launchers.
- `rl/` — RL data contracts, rewards, rollout partitioning, and envs.
- `moe/` — MoE runtime, losses, rank groups, and EP behavior.

## Local Checks

```bash
uv run pytest tests/config tests/distribution tests/evaluation tests/rl tests/moe
```

## GPU Smoke Tests

```bash
uv run pytest tests/e2e --dense --moe --vlm
```

## SLURM

```bash
tests/e2e/slurm/submit_e2e_tests.sh --dry-run
tests/e2e/slurm/submit_e2e_tests.sh
sbatch tests/e2e/fixtures/toy_async_eval_sidecar.sh
sbatch tests/e2e/fixtures/toy_async_eval_reserved.sh
```
