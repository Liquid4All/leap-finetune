# leap-finetune

A minimal fine-tuning repo for LFM2, fully built on Open Source.

> **⚠️ Important**
>
> - **Hardware:** We tested this tool on H100 80GB GPU. Multi-GPU parallelization has been tested up to 8 such GPUs.
> - **Operating system:** This tool currently supports Linux machines with the x86_64 architecture.
> - **Python:** Make sure you are running Python >= 3.12.
> - **Access token:** Make sure you are logged in on Hugging Face to access models and datasets.

For feature requests or if you have a different setup, reach out to [support@liquid.ai](mailto:support@liquid.ai) and tell us about your specific configuration.

## 🔧 Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone Repo

```bash
git clone <repository-url>
cd leap_finetune
```

### 3. Set up virtual environment

```bash
uv sync
```

## 🚀 Quickstart

### 1. Job Configuration Setup

Create a YAML config file (or copy one from [`job_configs/`](./job_configs/)):

```yaml
project_name: "my_sft_project"
model_name: "LFM2-1.2B"
training_type: "sft"

dataset:
  path: "HuggingFaceTB/smoltalk"
  type: "sft"
  limit: 1000
  test_size: 0.2
  subset: "all"

training_config:
  extends: "DEFAULT_SFT"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 2e-5

peft_config:
  extends: "DEFAULT_LORA"
  use_peft: true
```

- `training_config.extends` inherits from a base config (e.g. `DEFAULT_SFT`, `DEFAULT_DPO`, `DEFAULT_VLM_SFT`) — any fields you specify override the base
- `peft_config.extends` works the same way (e.g. `DEFAULT_LORA`, `DEFAULT_VLM_LORA`)
- See [`job_configs/`](./job_configs/) for more examples (DPO, MoE, VLM, SLURM)

### 2. Launch Training

Run locally:

```bash
uv run leap-finetune <path_to_config.yaml>
```

It uses Ray Train + Accelerate for distributed training.

Unless you overwrote `output_dir`, results will be stored in `outputs/training_type/job_name/`

### SLURM Support

If your config includes a `slurm` section, running `leap-finetune` will auto-generate and submit a SLURM script. You can also generate SLURM scripts without submitting:

```bash
uv run leap-finetune slurm <path_to_config.yaml>
```

To monitor your SLURM jobs in a TUI:

```bash
uv run turm --me
```

### 3. (Optional) Experiment Tracking with Weights & Biases

Set `wandb_logging: true` in your YAML config's `training_config` section. By default logs are saved locally to `./wandb/`. To sync to [wandb.ai](https://wandb.ai), set `WANDB_API_KEY`:

```bash
export WANDB_API_KEY=your_api_key
```

### 4. Bundle Checkpoint for LEAP

When training is done, you can bundle your output checkpoint with `leap-bundle` to use it directly within LEAP. Checkout our [Quick Start guide](https://leap.liquid.ai/docs/leap-bundle/quick-start?utm_source=github&utm_medium=link&utm_campaign=LEAP&utm_content=general).

## 📊 Expected Dataset Formats

### SFT (Supervised Fine-Tuning)

```json
{
  "messages": [
    { "role": "user", "content": "What is the capital of France?" },
    { "role": "assistant", "content": "The capital of France is Paris." }
  ]
}
```

### DPO (Direct Preference Optimization)

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "The capital of France is London."
}
```

### VLM SFT (Vision-Language Model)

```json
{
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are an image-based assistant. Answer questions based on the provided image."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        { "type": "image", "image": "/path/to/image.jpg" },
        { "type": "text", "text": "What do you see in this image?" }
      ]
    },
    {
      "role": "assistant",
      "content": [{ "type": "text", "text": "I see a car in the image." }]
    }
  ]
}
```

> **Note**: VLM datasets commonly have images in a separate row and are referenced in the messages column. If your image URLs or Paths are in a separate column from your messages, you'll need to merge the images into the 'messages' section like above.

## 🔄 Resuming Training

If a run is interrupted (SLURM preemption, crash, etc.), you can resume from the last checkpoint with full optimizer state, LR schedule, and wandb continuity.

Add `resume_from_checkpoint` to your `training_config`:

```yaml
training_config:
  resume_from_checkpoint: "latest"   # resumes from the most recent checkpoint
```

This resolves the `latest` symlink in your output directory (e.g. `outputs/my_project/latest → checkpoint-step-16000`). To resume from a specific checkpoint instead:

```yaml
training_config:
  resume_from_checkpoint: "/path/to/outputs/my_project/checkpoint-step-8000"
```

**What gets restored:** model weights, optimizer states, LR scheduler position, training step counter, and RNG states.

**Wandb continuity:** The wandb run ID is saved to `<output_dir>/.wandb_run_id` automatically. On resume, it appends metrics to the same run.

## 📈 Evaluation Benchmarks

Run benchmarks automatically during training at every `eval_steps`. Add a `benchmarks` section to your YAML config:

```yaml
benchmarks:
  max_new_tokens: 128
  benchmarks:
    - name: "mmmu_val"
      path: "/data/mmmu_val.jsonl"
      metric: "short_answer"

    - name: "imagenette"
      path: "/data/imagenette_eval.jsonl"
      metric: "logprob_zero_shot"
```

Benchmark data uses the **same format as training data** (HF messages schema). Available metrics: `short_answer`, `grounding_iou`, `mcq_gen`, `logprob_zero_shot`. Results are logged to wandb at `benchmark/{name}/score`.

See the [Evaluation Guide](./src/leap_finetune/evaluation/README.md) for data format examples, YAML reference, and how to add custom metrics.

## 🧪 Advanced Configuration

Default base configs live in [`src/leap_finetune/training_configs/`](./src/leap_finetune/training_configs/) and are auto-discovered — new configs added to these files are immediately available via `extends` in YAML.

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is pre-installed. Enable it with `use_liger_kernel: true` in your `training_config`.

## 📂 Dataset Loading

The `dataset.path` field in your YAML config accepts local files, HuggingFace Hub IDs, and cloud storage URIs:

| Source          | Example `path`                                 |
| --------------- | ---------------------------------------------- |
| Local file      | `/path/to/data.jsonl`, `/path/to/data.parquet` |
| HuggingFace Hub | `HuggingFaceTB/smoltalk`                       |
| S3              | `s3://bucket/path/to/data.parquet`             |
| GCS             | `gs://bucket/path/to/data.parquet`             |
| Azure           | `az://container/path/to/data.parquet`          |

Cloud storage requires appropriate credentials (AWS, GCP, or Azure). Use `subset` for HuggingFace datasets with multiple configs, and `limit` to cap the number of samples for quick testing.

## Contributing

1. Hook `pre-commit` to git: `uv run pre-commit install`
2. Open a PR with your changes

Pre-commit will now run automatically on commits, or run manually:

```bash
uv run pre-commit run --all-files
```

Please include a thorough description of changes and additions in your PR.
