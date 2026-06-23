---
name: lft-runs
description: Inspect and update leap-finetune local run state and experiment memory. Use for .lft/state.json, .lft/memory.md, run status, backend IDs, and agent handoff summaries.
---

# LFT Runs

## Use When

- Checking what training or eval runs have happened.
- Finding the latest run ID, status, backend, output path, or eval metrics.
- Syncing backend status for a submitted run.
- Adding a short experiment memory note that explains judgment, intent, or
  follow-up reasoning not already captured in structured state.

## Do Not Use When

- Creating or changing training configs. Use `lft-training`.
- Creating eval suites or metrics. Use `lft-evals`.
- Submitting or debugging remote backend code. Use `lft-launch`.

## Canonical Files

- `.lft/state.json`
- `.lft/memory.md`
- `src/leap_finetune/state/`
- `src/leap_finetune/cli/main.py`

## Steps

1. Start with `uv run leap-finetune runs report`.
2. For a specific run, use `uv run leap-finetune runs show <run_id>` or
   `uv run leap-finetune runs show latest`.
3. For remote-backed runs, use `uv run leap-finetune runs sync <run_id>` before
   reporting status.
4. Treat `.lft/state.json` as factual state: run ID, status/phase, heartbeat,
   step, latest log/eval/checkpoint, bounded metric history, backend metadata,
   output paths, and log refs.
5. Use `.lft/memory.md` only for non-discrete reasoning: why an experiment was
   tried, what result means, what to try next, or why a run should be ignored.
6. When adding memory, reference the run or eval ID instead of duplicating its
   metrics or metadata:

```bash
uv run leap-finetune memory add "Lower LR looked more stable; next try 5e-6 with same eval gate." --ref <run_id>
```

## Expected Output

- A concise run status summary with run IDs and backend IDs.
- Latest progress/eval/checkpoint facts from state before reading raw logs.
- A referenced memory note only when judgment or follow-up reasoning is needed.
- No duplicated config snapshots, metrics tables, or copied scheduler logs in
  `memory.md`.

## Verification

```bash
uv run leap-finetune runs list
uv run leap-finetune runs report
uv run leap-finetune memory show
```
