# Evaluation Benchmarks

Run benchmarks automatically during training at every `eval_steps`. Results are logged to wandb and printed to stdout. No code changes needed — configure everything in YAML.

## Quick Setup

Add a `benchmarks` section to your job config:

```yaml
training_config:
  eval_strategy: "steps"
  eval_steps: 2000

benchmarks:
  max_new_tokens: 128          # default for all generation benchmarks
  image_root: "/data/images"   # optional, prepended to relative image paths
  benchmarks:
    - name: "mmmu_val"
      path: "/data/mmmu_val.jsonl"
      metric: "short_answer"
      max_new_tokens: 50       # override per benchmark

    - name: "imagenette"
      path: "/data/imagenette_eval.jsonl"
      metric: "logprob_zero_shot"
```

That's it. The callback loads data once on the first eval step and caches it for all subsequent runs.

## Data Format

Benchmark data uses the **same format as training data** (HF messages schema). Supported file formats: JSONL, JSON, Parquet, CSV.


### Generation benchmarks

The last assistant turn is the ground truth. Everything before it is the prompt.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/img.jpg"},
        {"type": "text", "text": "What is shown in this image?"}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "A dog sitting on a beach."}]
    }
  ]
}
```

### Logprob MCQ benchmarks

No assistant turn needed. The model scores each option by log-probability and picks the highest.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/img.jpg"},
        {"type": "text", "text": "What animal is in this image?"}
      ]
    }
  ],
  "options": ["cat", "dog", "bird"],
  "answer_id": 1
}
```

### Text-only (no images)

```json
{
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "Capital of France?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Paris"}]}
  ]
}
```

## Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `short_answer` | generation | Case-insensitive substring match. Set `match_mode: "any_in_array"` if ground truth is a JSON array of acceptable answers. |
| `grounding_iou` | generation | Bounding-box IoU. Set `iou_threshold` (default 0.5). |
| `mcq_gen` | generation | Extracts MCQ letter (A-F) from generated text and compares to ground truth. |
| `logprob_zero_shot` | logprob | Zero-shot MCQ via per-option log-probability comparison. No generation needed. |

## YAML Reference

Per-benchmark fields:

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `name` | yes | — | Benchmark name (used in wandb keys: `benchmark/{name}/score`) |
| `path` | yes | — | Path to data file (JSONL, JSON, Parquet, CSV) |
| `metric` | yes | — | One of the metrics above |
| `max_new_tokens` | no | 128 | Max tokens to generate (generation metrics only) |
| `match_mode` | no | `"contains"` | For `short_answer`: `"contains"` or `"any_in_array"` |
| `iou_threshold` | no | 0.5 | For `grounding_iou` |
| `limit` | no | all | Cap number of eval samples |
| `format` | no | auto-detect | Force file format: `jsonl`, `json`, `parquet`, `csv` |
| `image_root` | no | — | Prepend to relative image paths |

## Adding a New Metric

1. Add a scoring function to `metrics.py`:
   ```python
   def score_my_metric(prediction: str, ground_truth: str, **_) -> float:
       # Return a float score for this sample
       ...
   ```

2. Register it in `_METRIC_DISPATCH` in the same file:
   ```python
   _METRIC_DISPATCH = {
       ...
       "my_metric": score_my_metric,
   }
   ```

3. Add the metric name to `GENERATION_METRICS` or `LOGPROB_METRICS` in `vlm_config.py`.

## How It Works

1. The trainer reads `benchmarks:` from your YAML config
2. `create_vlm_benchmarks_from_config()` creates benchmark objects based on the `metric` field
3. A `BenchmarkEvalCallback` is added to the HuggingFace Trainer
4. At every `eval_steps`, the callback:
   - Loads and normalizes data on first run (cached for subsequent evals)
   - Shards samples across GPUs
   - Runs evaluation (generation or logprob)
   - All-reduces scores across ranks
   - Logs averaged metrics to wandb
