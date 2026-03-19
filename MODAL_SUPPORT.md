# Modal Backend Support

## How it works

Adding a `modal:` section to any existing config YAML is all that's needed. The existing `leap-finetune` command auto-detects it, exactly like the `slurm:` key works today.

```bash
leap-finetune job_configs/sft_example_modal.yaml
```

```
leap-finetune config.yaml        (your machine)
        │
        ├── has slurm: key? → submit via sbatch
        ├── has modal: key? → submit to Modal        ← new
        └── neither         → run locally with Ray

Modal container (e.g. H100 x 4):
  ray.init(address="local")   ← discovers all 4 GPUs
  TorchTrainer → sft_run / dpo_run / vlm_sft_run
```

Modal provides the GPU machine on demand. Ray runs inside it unchanged. No changes to the training stack are needed.

---

## Changes required

### 1. Two new files

**`src/leap_finetune/backends/__init__.py`** — empty, makes it a package.

**`src/leap_finetune/backends/modal_backend.py`** — see implementation below.

### 2. Two lines added to `__init__.py`

Right after the existing SLURM check (line 118):

```python
    if check_and_handle_slurm(config_path_arg):
        return

    # Add these two lines:
    from leap_finetune.backends.modal_backend import check_and_handle_modal
    if check_and_handle_modal(config_path_arg):
        return
```

### 3. One additive change to `pyproject.toml`

```toml
[project.optional-dependencies]
modal = ["modal>=0.73"]
```

Install with: `uv sync --extra modal`

### 4. One new example config

**`job_configs/sft_example_modal.yaml`** — see example below.

---

## `modal_backend.py` implementation

```python
import os
import sys

import yaml


def check_and_handle_modal(config_path_arg: str) -> bool:
    """Mirrors check_and_handle_slurm(). Returns True if Modal job was submitted."""
    if not config_path_arg:
        return False

    try:
        from leap_finetune.utils.config_parser import resolve_config_path
        config_path = resolve_config_path(config_path_arg)
    except Exception:
        return False

    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        modal_cfg = config_dict.get("modal")
        if not modal_cfg:
            return False

        print("Config contains Modal settings - submitting Modal job...")
        _submit(config_dict, modal_cfg)
        return True
    except Exception as e:
        print(f"Error submitting Modal job: {e}")
        return False


def _submit(config_dict: dict, modal_cfg: dict) -> None:
    import modal

    # Strip modal: key so the container doesn't re-dispatch
    config_dict = {k: v for k, v in config_dict.items() if k != "modal"}

    # Point output_dir at the mounted volume so checkpoints are persisted
    output_dir = modal_cfg.get("output_dir", "/outputs")
    config_dict.setdefault("training_config", {})["output_dir"] = output_dir

    config_str = yaml.dump(config_dict)

    app = modal.App(modal_cfg.get("app_name", "leap-finetune"))
    image = _build_image(modal_cfg.get("base_image", "nvcr.io/nvidia/pytorch:25.02-py3"))
    volume = modal.Volume.from_name(
        modal_cfg.get("output_volume", "leap-finetune-outputs"), create_if_missing=True
    )
    secrets = [modal.Secret.from_name(s) for s in modal_cfg.get("secrets", [])]

    @app.function(
        image=image,
        gpu=modal_cfg.get("gpu", "H100"),
        timeout=modal_cfg.get("timeout", 86400),
        volumes={output_dir: volume},
        secrets=secrets,
    )
    def _train(cfg: str) -> None:
        import sys
        import tempfile
        from leap_finetune import main as leap_main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(cfg)
            tmp_path = f.name

        sys.argv = ["leap-finetune", tmp_path]
        leap_main()

    with app.run():
        _train.remote(config_str)


def _build_image(base_image: str):
    import modal

    return (
        modal.Image.from_registry(base_image)
        .pip_install(
            "transformers>=5.0.0",
            "peft>=0.15.2",
            "accelerate>=1.7.0",
            "trl>=0.18.2",
            "ray==2.48.0",
            "deepspeed>=0.17.1",
            "liger-kernel>=0.6.2",
            "datasets",
            "pyyaml>=6.0",
            "wandb>=0.22.3",
            "torchvision>=0.24.1",
            "pillow>=11.3.0",
            "rich>=14.1.0",
            "psutil",
        )
        .pip_install("flash-attn>=2.8.0", extra_options="--no-build-isolation")
        .add_local_python_source("leap_finetune")
    )
```

`.add_local_python_source("leap_finetune")` bundles the local `src/leap_finetune/` source into the container at submission time, so the exact version you have locally is what runs — no publish step needed.

---

## Example config: `job_configs/sft_example_modal.yaml`

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

modal:
  app_name: "leap-finetune"
  gpu: "H100:4" # 4x H100 — Ray creates one worker per GPU
  timeout: 86400 # 24 hours
  output_volume: "leap-finetune-outputs" # Modal Volume, created automatically
  output_dir: "/outputs" # mount point inside the container
  secrets:
    - "huggingface-secret" # must contain HF_TOKEN
    # - "wandb-secret"                 # optional, must contain WANDB_API_KEY
```

**GPU options:**

| `gpu:` value                | Hardware                      |
| --------------------------- | ----------------------------- |
| `"H100"`                    | 1x H100 80GB                  |
| `"H100:4"`                  | 4x H100 80GB                  |
| `"H100:8"`                  | 8x H100 80GB                  |
| `"A100-80GB:4"`             | 4x A100 80GB                  |
| `["H100:4", "A100-80GB:4"]` | H100 preferred, A100 fallback |

---

## User setup (one-time)

```bash
# Install Modal and authenticate
pip install modal
modal setup

# Create secrets for API tokens
modal secret create huggingface-secret HF_TOKEN=hf_...
modal secret create wandb-secret WANDB_API_KEY=...   # optional

# Install leap-finetune with modal extra
uv sync --extra modal
```

## Running a job

```bash
leap-finetune job_configs/sft_example_modal.yaml
```

Checkpoints are saved to the Modal Volume and visible at https://modal.com/storage. To download them locally:

```bash
modal volume get leap-finetune-outputs /outputs ./local-outputs
```

---

## Constraints

- **Datasets must be on HuggingFace Hub or cloud storage** (S3, GCS, Azure). Local file paths are not accessible inside the Modal container. Upload local datasets to HF Hub first with `datasets.Dataset.push_to_hub()`.
- **Models must be on HuggingFace Hub.** `model_name: "LFM2-1.2B"` resolves to `LiquidAI/LFM2-1.2B` and downloads at job start. Local model paths do not work.
- **Single-node only.** Same limitation as the existing SLURM backend. Up to 8 GPUs on one machine.
- **First run is slow.** The image (CUDA base + all deps) takes ~15 minutes to build on first use. Modal caches it after that.
