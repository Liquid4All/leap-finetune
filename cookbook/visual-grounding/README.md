# Visual Grounding cookbook (LFM2.5-VL)

End-to-end recipe for teaching LFM2.5-VL to localize objects in images
and emit bounding boxes in the model's native JSON format:

```json
[{ "label": "red car", "bbox": [0.12, 0.34, 0.58, 0.71] }]
```

with normalized `[0, 1]` coordinates.

The recipe runs in two phases:

| Phase                | Goal                                                                | Dataset                                                                                                              | This file                                |
| -------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **1. SFT**           | Teach the JSON bbox format from ~500K supervised examples.          | [Michael4933/MGrounding-630k](https://huggingface.co/datasets/Michael4933/MGrounding-630k) (minus `Object_Tracking`) | `configs/sft_grounding.yaml`             |
| **2. GRPO** _(next)_ | Refine with a Hungarian-matched IoU-F1 reward — multi-object aware. | Subset of phase 1 data                                                                                               | `configs/grpo_grounding.yaml` _(coming)_ |

Both phases use async eval (sidecar mode) so training never pauses for
benchmark scoring. Eval suite:

- **RefCOCO val** — canonical single-box IoU, comparable to published
  Qwen2.5-VL / PaliGemma / Molmo numbers.
- **MIG-Bench** — Migician's official multi-image grounding benchmark
  (ACL 2025), in-distribution check for MGrounding-trained models.

---

## Phase 1 — SFT

### 0. Pre-reqs

```bash
# From repo root
uv sync
huggingface-cli login              # for MGrounding-630k download
wandb login                        # optional, for live training curves
```

### 1. Build the SFT + GRPO + test parquets (one-time, ~140 GB download)

**Always run as an sbatch job — the head node will not survive the
extraction.** Compute is CPU-only (no GPU needed):

```bash
sbatch cookbook/visual-grounding/configs/prepare_data.sh
```

What it does:

1. `snapshot_download Michael4933/MGrounding-630k` (~140 GB)
2. Extracts subset zips. `Group_Grounding` is multi-volume — needs
   `7z` (`apt install p7zip-full`) since Python's stdlib `zipfile`
   can't read spanned archives. `Object_Tracking` is downloaded but
   **not extracted** (we filter it at conversion).
3. Parses the manifest, walks multi-turn conversations, normalizes
   coordinates, then **deterministically splits 3-way**:

This pulls `Michael4933/MGrounding-630k` from HuggingFace, drops the
`Object_Tracking` subset, walks each multi-turn conversation extracting
each (human, gpt) pair with bboxes, converts MGrounding's
`<|box_start|>(x,y),(x,y)<|box_end|>` 0-1000 coords into normalized
`[0, 1]` xyxy. Multi-image inputs (2-5 images/sample) are preserved.
Then **deterministically shuffles and splits into two non-overlapping
pools**:

- `<output>/grounding_sft/train.parquet` — 65% (~1.3M output rows)
- `<output>/grounding_grpo/train.parquet` — 25% (~500K output rows)
- `<output>/grounding_test/test.parquet` — 10% (~200K output rows)

(Counts are higher than MGrounding's 630K input rows because each
multi-turn conversation expands into one output per qualifying
turn-pair.)

All three pools are **pairwise disjoint** (seeded shuffle). GRPO
never sees SFT data, so the model can't memorize during RL fine-tuning.
The test pool is held out from both training phases and plugged into
the YAML's `benchmarks` block — it's the in-distribution grounding
curve in wandb alongside the canonical RefCOCO trio.

All three files share the same single-column `messages` schema —
leap-finetune's `vlm_grpo` loader auto-extracts `prompt` and
`solution` from `messages` at load time. **One conversion, three uses.**

### 2. Build async-eval files (RefCOCO trio)

```bash
sbatch cookbook/visual-grounding/configs/prepare_evals.sh
```

Lightweight CPU job (~10 min) that downloads RefCOCO val + RefCOCO+
val + RefCOCOg val from HuggingFace and emits 500-sample jsonls each.

Writes `refcoco_val.jsonl` and `mig_bench.jsonl` (500 samples each by
default — enough for fast async eval cycles every `eval_steps=2000`).

### 3. Launch SFT

Single-node (8 GPUs):

```bash
sbatch cookbook/visual-grounding/configs/sft_grounding.sh
```

Or interactively:

```bash
uv run leap-finetune cookbook/visual-grounding/configs/sft_grounding.yaml
```

Training writes checkpoints to `outputs/visual_grounding_sft/{run}/`.
Wandb shows:

- `train/loss`
- `benchmark/refcoco_val/score` — single-box IoU@0.5 on RefCOCO val
- `benchmark/mig_bench/score` — single-box IoU@0.5 on MIG-Bench

All three curves share the same `train/global_step` X axis — async eval
results are back-filled to the originating training step.

---

## Data format

The cookbook's `prepare_data.py` emits one row per sample with this
schema (matches every `vlm_sft` / `vlm_grpo` recipe in leap-finetune):

```python
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "image": "/abs/path/to/image.jpg"},
      {"type": "text",  "text": 'Identify the region of "red car" ...'},
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": '[{"label": "red car", "bbox": [0.12, 0.34, 0.58, 0.71]}]'},
    ]},
  ]
}
```

If you bring your own grounding dataset, match this schema and the
existing YAML drops in with only `dataset.path` updated. No code
changes needed for new datasets.

---

## Customer adaptation

To swap MGrounding for your own dataset, you only need to:

1. Pre-process your data to the `messages` schema above (one
   parquet row = one conversation; bbox coords normalized 0-1).
2. Point `dataset.path` at your parquet directory in the YAML.
3. _(Optional)_ Update `benchmarks[*].path` to your eval jsonl.

The reward, metric, callback, async-eval, and slurm wiring all stay
the same.

---

## What's next

- `configs/grpo_grounding.yaml` — Phase 2 trains on the held-out
  `./job_datasets/grounding/grounding_grpo/train.parquet` (already
  produced by step 1 above) using the existing
  `VLMGroundingIoURecipe`: strict-format reward (0.1) + Hungarian-matched
  IoU-F1 reward (1.0). The GRPO model starts from the SFT checkpoint
  and learns to recover IoU points lost to greedy decoding.
