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

Go to [`config.py`](./config.py) and follow the instructions there.

- Use `DatasetLoader` to load datasets from HuggingFace Hub or local files (you can also add custom data loading logic here as long as it's TRL compatible)
- Pick a default `TrainingConfig` and optionally override some of the config parameters. Pick a `PeftConfig`.
- Create a `JobConfig` with your desired settings (model, dataset, etc.)

### 2. Launch Training

Run locally:

```bash
uv run leap-finetune
```

It uses Ray Train + Accelerate for distributed training.

Unless you overwrote `output_dir`, results will be stored in `outputs/training_type/job_name/`

### 3. (Optional) Experiment Tracking with Weights & Biases

To enable experiment tracking (using [Weights & Biases](https://wandb.ai)):

- Set `wandb_logging=True` in `config.py` in your `user_config` overrides or default configs.
- **Offline mode (default)**: If no `WANDB_API_KEY` is set, wandb logs locally to `./wandb/` directory. No API key needed!
- **Online mode**: Set the `WANDB_API_KEY` environment variable to sync to wandb.ai dashboard:

```bash
export WANDB_API_KEY=your_api_key  # optional; for online syncing to wandb.ai
```

You can also customize the project name (defaults to `"leap-finetune"`):

```bash
export WANDB_PROJECT=my-custom-project  # optional; defaults to "leap-finetune"
```

After training, view your metrics:

- **Online mode**: View at `https://wandb.ai/<your-entity>/<project-name>/runs/<run-name>`
- **Offline mode**: Sync later with `wandb sync ./wandb/offline-run-*` or view locally

Runs are named after your `job_name` and metrics are reported via TRL/Transformers. Training metrics (loss, learning rate, etc.) are logged every `logging_steps` (default: 10), and evaluation metrics are logged at the end of each epoch.

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

## 🧪 Advanced Configuration

### Default Configs Location and Adding New Configs

The default configurations are located in:

- **SFT Training**: [`src/leap_finetune/configs/sft_configs.py`](./src/leap_finetune/configs/sft_configs.py)
- **DPO Training**: [`src/leap_finetune/configs/dpo_configs.py`](./src/leap_finetune/configs/dpo_configs.py)
- **PEFT/LoRA**: [`src/leap_finetune/configs/peft_configs.py`](./src/leap_finetune/configs/peft_configs.py)

To add a new training configuration add it to the respective file and then reference it in [`src/leap_finetune/configs/__init__.py`](./src/leap_finetune/configs/__init__.py) in the `TrainingConfig` and/or `PeftConfig` enum.

We also support [Liger Kernel](https://github.com/linkedin/Liger-Kernel) and it comes pre-installed.
Just add `"use_liger_kernel": True"` to your `user_config`

## 📂 Advanced Dataset Loading

`DatasetLoader` supports multiple data sources with automatic format detection and validation.

### DatasetLoader Parameters

| Parameter      | Type                              | Default   | Description                                            |
| -------------- | --------------------------------- | --------- | ------------------------------------------------------ |
| `dataset_path` | `str`                             | required  | Path to dataset (local, cloud, or HuggingFace Hub ID)  |
| `dataset_type` | `"sft"` \| `"dpo"` \| `"vlm_sft"` | required  | Training format type                                   |
| `limit`        | `int`                             | `None`    | Limit number of samples (useful for testing)           |
| `split`        | `str`                             | `"train"` | Dataset split to use                                   |
| `test_size`    | `float`                           | `0.2`     | Fraction of data for evaluation                        |
| `subset`       | `str`                             | `None`    | Dataset subset (for HuggingFace datasets with configs) |

### Supported Data Sources

#### Local Files

```python
# JSONL file
DatasetLoader("/path/to/data.jsonl", "sft")

# Parquet file (faster for large datasets)
DatasetLoader("/path/to/data.parquet", "sft")
```

#### HuggingFace Hub

```python
# Public dataset
DatasetLoader("HuggingFaceTB/smoltalk", "sft", subset="all")

# Private dataset (requires HF login)
DatasetLoader("your-org/private-dataset", "sft")
```

#### Cloud Storage

Requires appropriate credentials configured (AWS credentials, GCP service account, Azure credentials).

```python
# Amazon S3
DatasetLoader("s3://bucket/path/to/data.parquet", "sft")
DatasetLoader("s3://bucket/path/to/data.jsonl", "sft")

# Google Cloud Storage
DatasetLoader("gs://bucket/path/to/data.parquet", "sft")

# Azure Blob Storage
DatasetLoader("az://container/path/to/data.parquet", "sft")
DatasetLoader("abfs://container@account.dfs.core.windows.net/path/data.parquet", "sft")
```

### Quick Testing with Limits

Use `limit` to quickly test your pipeline with a subset of data:

```python
# Test with 100 samples
DatasetLoader("HuggingFaceTB/smoltalk", "sft", subset="all", limit=100)

# Full dataset
DatasetLoader("HuggingFaceTB/smoltalk", "sft", subset="all")
```

### File Format Recommendations

| Format          | Best For                         | Notes                                          |
| --------------- | -------------------------------- | ---------------------------------------------- |
| **Parquet**     | Large datasets (>100K rows)      | Columnar format, fast reads, smaller file size |
| **JSONL**       | Smaller datasets, human-readable | Line-delimited JSON, easy to inspect           |
| **HuggingFace** | Public datasets                  | Automatic streaming, no local storage needed   |

### Custom Preprocessing

For datasets that need reformatting, filtering, or joining before training, use the `preprocess_fn` parameter. This function receives a Ray Dataset and must return a Ray Dataset in the expected format.

```python
import ray.data

def my_preprocess(ds: ray.data.Dataset) -> ray.data.Dataset:
    """Custom preprocessing - runs before validation."""

    # Example: Filter rows where content length > 100
    ds = ds.filter(lambda row: len(row.get("content", "")) > 100)

    # Example: Transform column names
    ds = ds.map(lambda row: {
        "messages": [
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]}
        ]
    })

    # Example: Sample 10% of data
    ds = ds.random_sample(0.1)

    return ds

# Use with DatasetLoader
DatasetLoader(
    "path/to/raw-data.jsonl",
    "sft",
    preprocess_fn=my_preprocess
)
```

**Common preprocessing operations:**

| Operation       | Ray Data Method                                      |
| --------------- | ---------------------------------------------------- |
| Filter rows     | `ds.filter(lambda row: condition)`                   |
| Transform rows  | `ds.map(lambda row: new_row)`                        |
| Batch transform | `ds.map_batches(fn, batch_format="pandas")`          |
| Sample data     | `ds.random_sample(fraction)`                         |
| Drop columns    | `ds.drop_columns(["col1", "col2"])`                  |
| Rename columns  | `ds.map(lambda row: {new_name: row[old_name], ...})` |

See [Ray Data documentation](https://docs.ray.io/en/latest/data/api/dataset.html) for all available operations.

## Contributing

1. Hook `pre-commit` to git: `uv run pre-commit install`
2. Open a PR with your changes

Pre-commit will now run automatically on commits, or run manually:

```bash
uv run pre-commit run --all-files
```

Please include a thorough description of changes and additions in your PR.
