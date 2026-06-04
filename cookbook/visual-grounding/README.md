# Visual Grounding cookbook (LFM2.5-VL)

End-to-end recipe for teaching LFM2.5-VL to localize objects in images
and emit bounding boxes in the model's native JSON format:

```json
[{ "label": "red car", "bbox": [0.12, 0.34, 0.58, 0.71] }]
```

with normalized `[0, 1]` coordinates.

Two phases:

| Phase      | Goal                                                       | This file                     |
| ---------- | ---------------------------------------------------------- | ----------------------------- |
| **1. SFT** | Teach the JSON bbox format from MGrounding-630k.           | `configs/sft_grounding.yaml`  |
| **2. GRPO**| Refine with Hungarian-matched CIoU-F1 reward.              | `configs/grpo_grounding.yaml` |

Both phases use async eval (sidecar mode) so training never pauses for
benchmark scoring. Eval suite:

- **RefCOCO val / RefCOCO+ val / RefCOCOg val** — canonical single-box
  IoU, comparable to published Qwen2.5-VL / PaliGemma / Molmo numbers.
- **mgrounding_test** — 10% held-out slice of MGrounding-630k itself
  (multi-image, multi-bbox; in-distribution sanity check).

---

## Phase 1 — SFT

### 0. Pre-reqs

```bash
# From repo root
uv sync
huggingface-cli login              # MGrounding-630k is a HF dataset
wandb login                        # optional, for live training curves
```

### 1. Build the SFT + GRPO + test parquets (one-time, ~140 GB download)

CPU-only job — run as `sbatch` because the multi-volume zip extraction
will swamp a login node:

```bash
sbatch cookbook/visual-grounding/configs/prepare_data.sh
```

What `prepare_data.py` does:

1. `snapshot_download Michael4933/MGrounding-630k` (~140 GB).
2. Extracts subset zips. `Group_Grounding` and `Object_Tracking` are
   multi-volume — needs `7z` (`apt install p7zip-full`) since Python's
   stdlib `zipfile` can't read spanned archives.
3. Walks each conversation, expands `<|box_start|>(x,y),(x,y)<|box_end|>`
   coords in MGrounding's native 0-1000 space into normalized
   `[0, 1]` xyxy, and canonicalizes every variant (single-image,
   group-grounding, multi-image Object_Tracking) into the same
   `[{"label", "bbox"}]` JSON schema. Multi-image conversations (2-5
   images / sample) are preserved.
4. Deterministically shuffles and splits 3-way:

   - `<output>/grounding_sft/train.parquet` — ~72%
   - `<output>/grounding_grpo/train.parquet` — ~18%
   - `<output>/grounding_test/test.parquet`  — ~10%

   Pairwise disjoint (seeded shuffle). GRPO never sees SFT data, so the
   model can't memorize during RL fine-tuning. The test pool is held out
   from both training phases and plugged into the YAML's `benchmarks`
   block as `mgrounding_test`.

All three pools share the same single-column `messages` schema —
leap-finetune's `vlm_sft` and `vlm_grpo` loaders extract `prompt` and
`solution` from `messages` at load time. **One conversion, three uses.**

### 2. Build RefCOCO eval files

```bash
sbatch cookbook/visual-grounding/configs/prepare_evals.sh
```

Lightweight CPU job (~10 min). Downloads RefCOCO val + RefCOCO+ val +
RefCOCOg val from HF and emits one jsonl per benchmark (full val splits
by default: 3811 + 3805 + 2573 samples; ±1 IoU-point noise floor).

### 3. Launch SFT

Single-node (8 GPUs):

```bash
sbatch cookbook/visual-grounding/configs/sft_grounding.sh
```

Or 2 nodes × 8 GPUs:

```bash
sbatch cookbook/visual-grounding/configs/sft_grounding_multinode.sh
```

Or interactively:

```bash
uv run leap-finetune cookbook/visual-grounding/configs/sft_grounding.yaml
```

Wandb logs:

- `train/loss`
- `benchmark/refcoco_val/score`, `refcoco_plus_val/score`,
  `refcocog_val/score` — single-box IoU@0.5.
- `benchmark/mgrounding_test/score` — Hungarian-matched CIoU-F1
  (multi-box-aware).

All curves share the same `benchmark/step` axis — async eval results
are back-filled to the originating training step so curves don't get
clobbered by later training steps.

---

## Phase 2 — GRPO

Refines the SFT checkpoint on the 18% GRPO holdout using the shipped
`VLMGroundingCIoURecipe`:

- **strict_format reward** (weight 0.1) — the completion must parse as a
  JSON array of `{"label", "bbox"}` dicts.
- **ciou_f1 reward** (weight 1.0) — Hungarian-matches predicted boxes
  against ground truth (matcher uses CIoU so it prefers center-aligned,
  same-shape pairs), then scores the F1 of the matched CIoUs. F1
  naturally penalizes false positives (drags precision) and false
  negatives (drags recall). Reduces to a single CIoU when ground truth
  has one box.

If you'd rather use plain IoU (no center-distance or aspect-ratio
penalty), swap `VLMGroundingCIoURecipe` → `VLMGroundingIoURecipe` in
the YAML.

### 1. Point the GRPO YAML at your SFT checkpoint

Open `configs/grpo_grounding.yaml` and set `model_name` to the SFT
final checkpoint path:

```yaml
model_name: "./outputs/visual_grounding_sft/<your-sft-final-checkpoint>"
```

### 2. Launch

```bash
sbatch cookbook/visual-grounding/configs/grpo_grounding.sh
# or multinode:
sbatch cookbook/visual-grounding/configs/grpo_grounding_multinode.sh
```

Same four async-eval benchmarks as Phase 1 — wandb shows the RefCOCO
trio + `mgrounding_test` climbing as GRPO pushes the SFT prior toward
higher IoU under the strict-format constraint.

---

## Data format

`prepare_data.py` emits one row per sample with this schema (matches
every `vlm_sft` / `vlm_grpo` recipe in leap-finetune):

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

To swap MGrounding for your own grounding dataset:

1. Pre-process your data to the `messages` schema above (one parquet
   row = one conversation; bbox coords normalized to `[0, 1]`).
2. Point `dataset.path` at your parquet directory in the YAML.
3. _(Optional)_ Update `benchmarks[*].path` to your eval jsonl.

The reward, metric, callback, async-eval, and slurm wiring all stay
the same.
