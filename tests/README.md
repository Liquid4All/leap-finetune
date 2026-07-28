# Tests

Keep the suite intentionally small. New tests should land in one of three buckets:

- `config/` — Pydantic config parsing and validation.
- `math/` — numerical correctness for losses, metrics, masks, routing, and sharding.
- `e2e/` — full-service training tests, fixtures, and SLURM launchers.

## Local Checks

```bash
uv run pytest tests/config tests/math
```

On AMD / ROCm environments, use the ROCm project so CUDA and ROCm locks remain
separate. Set this once in your shell, module, or direnv config:

```bash
export UV_PROJECT=envs/rocm
uv run python -m pytest tests/config tests/math
```

## GPU Smoke Tests

```bash
uv run pytest tests/e2e --dense --moe --vlm --retrieval
```

For ROCm:

```bash
UV_PROJECT=envs/rocm uv run python -m pytest tests/e2e --dense --moe --vlm --retrieval
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
UV_PROJECT=envs/rocm uv sync
UV_PROJECT=envs/rocm uv run leap-finetune env fa2-status --require
```

## SLURM

```bash
tests/e2e/slurm/submit_e2e_tests.sh --dry-run
tests/e2e/slurm/submit_e2e_tests.sh
sbatch tests/e2e/fixtures/toy_async_eval_sidecar.sh
sbatch tests/e2e/fixtures/toy_async_eval_reserved.sh
```
