# Tests

Keep the suite intentionally small. New tests should land in one of these
buckets:

Do not add root-level `tests/test_*.py` files; use the bucketed directories
below.

Prefer regression tests over construction-time checks. Do not add tests that
only assert Pydantic validators, static default constants, or one-off behavior
that was useful while building a feature but is not useful long term.

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

On AMD / ROCm environments, use the ROCm project so CUDA and ROCm locks remain
separate. Set this once in your shell, module, or direnv config:

```bash
export UV_PROJECT=rocm
uv run python -m pytest tests/config tests/distribution tests/evaluation tests/rl tests/moe
```

## GPU Smoke Tests

```bash
uv run pytest tests/e2e --dense --moe --vlm
```

For ROCm:

```bash
UV_PROJECT=rocm uv run python -m pytest tests/e2e --dense --moe --vlm
```

## FA2 Validation

Normal tests may fall back to SDPA. FA2 validation should inspect the active
environment and can require runtime selection:

```bash
uv sync
uv run leap-finetune env fa2-status --require
```

For ROCm:

```bash
UV_PROJECT=rocm uv sync
UV_PROJECT=rocm uv run leap-finetune env fa2-status --require
```

## SLURM

```bash
tests/e2e/slurm/submit_e2e_tests.sh --dry-run
tests/e2e/slurm/submit_e2e_tests.sh
sbatch tests/e2e/fixtures/toy_async_eval_sidecar.sh
sbatch tests/e2e/fixtures/toy_async_eval_reserved.sh
```
