# Performance TODO

These are useful follow-ups after the first cleanup pass. They are intentionally
not required for the initial agent-skills PR.

## Remaining Items

- Reduce the remaining Ray/Hugging Face boundary materialization in
  `ray_dataset_to_hf`. Today each worker shard is converted to a Hugging Face
  `Dataset` for Trainer compatibility; investigate streaming or bounded-shard
  alternatives before touching the trainer contract.
- Remove the full `count()` required for ratio-based `dataset.test_size` splits.
  Explicit train/eval paths and no-split training now avoid it, but ratio splits
  still need a deterministic row count unless we switch to another split method.
- Replace Python `filter(fn=...)` Ray predicates with expression-based filters
  where possible to cut Python overhead and clear Ray's performance warning.
- Make tokenization cache defaults more automatic for repeated SFT/DPO
  hill-climb runs. The cache is atomic today, but `dataset.cache_dataset` is
  still opt-in.
- Tune worker dataloaders for throughput. Ray shards are converted to
  Hugging Face datasets, then read with plain `DataLoader` defaults; evaluate
  `num_workers`, `pin_memory`, `persistent_workers`, and prefetch settings.
- Continue MoE / expert-parallel hot-path cleanup. Likely wins include
  replacing `torch.histc` token counts with `bincount`, reducing sort/scatter
  allocations, preallocating EP buffers, and improving all-to-all overlap.
- Improve persistent eval and vLLM reuse. Async sidecar and GRPO server modes
  exist, but startup/reload overhead can still dominate large eval loops.
- Reduce checkpoint/export stalls. Investigate async checkpointing, parallel
  save paths, less staging duplication, and better defaults for shared storage.
