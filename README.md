<div align="center">
  <img
    src="./banner.png"
    alt="leap-finetune"
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em;">
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> •
    <a href="https://docs.liquid.ai/lfm"><strong>Documentation</strong></a> •
    <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a>
  </div>
  <br/>
  <a href="https://discord.com/invite/liquid-ai"><img src="https://img.shields.io/discord/1385439864920739850?style=for-the-badge&logo=discord&logoColor=white&label=Discord&color=5865F2" alt="Join Discord"></a>
</div>
</br>

<p align="center">
<a href="#-setup">Setup</a> · <a href="#-quickstart">Quickstart</a> · <a href="#-expected-dataset-formats">Dataset Formats</a> · <a href="#-tool-calling-datasets">Tool Calling</a> · <a href="#-resuming-training">Resuming Training</a> · <a href="#-evaluation-benchmarks">Benchmarks</a> · <a href="#-advanced-configuration">Advanced Config</a>
</p>

LEAP-Finetune is a minimal fine-tuning repo for LFM2, fully built on Open Source. It handles multi-gpu orchestration, dataset formatting and validation, and model checkpointing. We support different acceleration backends, including GPU nodes of 8xH100 80GB (both single node and multi node) as well as Modal (H100, H200, B200, ..) in case you don't have your own GPUs.

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

Unless you overwrote `output_dir`, results will be stored in `outputs/{project_name}/{run_name}/`. Each run gets its own directory with a unique name based on model, dataset, LR, and timestamp.

### Modal Support

You can run training jobs on Modal's serverless GPUs directly from your Mac or laptop — no local GPU required.

**One-time setup:**

```bash
huggingface-cli login   # required — used for model downloads and trackio
modal setup              # configure Modal credentials
```

**Add a `modal:` section to any config:**

```yaml
modal:
  gpu: "H100:4"
  timeout: 86400
  output_volume: "leap-finetune"
  output_dir: "/outputs"
  detach: false
```

**Run:**

```bash
uv run leap-finetune job_configs/sft_example_modal.yaml
```

That's it. The CLI will:

1. Build the container image (~5 min on first run, cached after that)
2. Auto-create a `huggingface-secret` on Modal from your local HF token
3. Stream build and training logs to your terminal in real-time
4. Save checkpoints to a Modal Volume

**Retrieving checkpoints:**

```bash
modal volume ls leap-finetune                                        # list saved checkpoints
modal volume get leap-finetune <checkpoint-name> ./local-outputs     # download to local
```

**Detached mode:** Set `detach: true` in the modal config to submit and disconnect. Monitor with `modal app logs leap-finetune`.

See [`job_configs/sft_example_modal.yaml`](./job_configs/sft_example_modal.yaml) for all available options.

### SLURM Support

If your config includes a `slurm` section, running `leap-finetune` will auto-generate and submit a SLURM script. You can also generate SLURM scripts without submitting:

```bash
uv run leap-finetune slurm <path_to_config.yaml>
```

To monitor your SLURM jobs in a TUI:

```bash
uv run turm --me
```

### External Ray Clusters

`leap-finetune` now supports attaching to an existing Ray cluster instead of forcing single-node local Ray.

You can provide the cluster address in either:

```bash
export RAY_ADDRESS=<head-ip>:6379
export LEAP_RAY_NUM_WORKERS=16
uv run leap-finetune <path_to_config.yaml>
```

or in config:

```yaml
ray:
  address: "<head-ip>:6379"
  num_workers: 16
  resources_per_worker:
    GPU: 1
```

Precedence is `LEAP_RAY_ADDRESS`, then `RAY_ADDRESS`, then `ray.address` in YAML. If no address is provided, `leap-finetune` starts a local single-node Ray runtime like before.

This makes non-SLURM environments such as Nebius or any custom launcher workable as long as they:

1. Start a Ray head node and workers however they want.
2. Export `RAY_ADDRESS` or `LEAP_RAY_ADDRESS` for the driver.
3. Optionally set `LEAP_RAY_NUM_WORKERS` if the cluster has more GPUs than the job should consume.

For convenience there is a generic helper script:

```bash
scripts/run_with_ray_cluster.sh <path_to_config.yaml>
```

### Multi-Node SLURM

Generated SLURM scripts now bootstrap a Ray cluster automatically when `slurm.nodes > 1`. They use [job_configs/slurms/utils/slurm_ray.sh](/home/alay/leap-finetune-24b/job_configs/slurms/utils/slurm_ray.sh) to:

1. Start the Ray head on the first allocated node.
2. Start Ray workers on the remaining nodes.
3. Wait for the cluster to register all GPUs.
4. Export `RAY_ADDRESS` before launching `leap-finetune`.

That keeps the single-node path simple while making the multi-node path explicit.

### Docker Path

A basic CUDA image is available in [Dockerfile](/home/alay/leap-finetune-24b/Dockerfile). It is meant for the same external-cluster contract: the container only needs to receive `RAY_ADDRESS` and, optionally, `LEAP_RAY_NUM_WORKERS`.

Example:

```bash
docker build -t leap-finetune:latest .
docker run --rm --gpus all --network host \
  -e RAY_ADDRESS=<head-ip>:6379 \
  -e LEAP_RAY_NUM_WORKERS=16 \
  -v "$(pwd)":/workspace \
  leap-finetune:latest job_configs/long_context_moe_sft_example.yaml
```

For orchestration systems like Tangle, the important part is not Docker specifically; it is that the driver process can see the repo and gets the Ray address injected. The same contract works in bare-metal, VM, or container launchers.

### KubeRay / Kubernetes

`leap-finetune` can also submit a KubeRay `RayJob` directly when the YAML config contains a `kuberay:` section.

This path is built on the same external-Ray contract described above. The KubeRay helper does two things:

1. It creates a `RayCluster` spec with a head pod plus a fixed GPU worker pool.
2. It writes the resolved training config into a ConfigMap and injects `ray.address: "auto"` and `ray.num_workers` so the driver attaches to that cluster instead of starting local Ray.

That means the image belongs in the KubeRay pod templates, not in per-node imperative setup. For orchestrators like Tangle, the Kubernetes-native options are:

1. Have Tangle create a `RayJob` directly.
2. Have Tangle manage a persistent `RayCluster` and run `leap-finetune` with `RAY_ADDRESS` or `LEAP_RAY_ADDRESS`.

Minimal example:

```yaml
kuberay:
  image: "your-registry.com/leap-finetune:latest"
  namespace: "default"
  worker_replicas: 2
  gpus_per_worker: 4
  head_cpu: 4
  head_memory: "16Gi"
  worker_cpu: 8
  worker_memory: "64Gi"
  output_pvc: "training-outputs"
```

Then submit with the normal CLI:

```bash
uv run leap-finetune your_config_with_kuberay.yaml
```

That path creates a ConfigMap for the resolved training config, submits a `RayJob`, and exits after printing the `kubectl` commands to monitor it. It supports both:

1. Non-Docker workflows, where you clone the repo locally, run `uv sync`, and point `kubectl` at an existing cluster.
2. Docker workflows, where the `kuberay.image` points at a built image from [Dockerfile](/home/alay/leap-finetune-24b/Dockerfile) or an equivalent image that already contains `leap-finetune`.

The dispatcher first tries local kubeconfig and then in-cluster config, so the same code works from a developer machine, a CI runner, or a control pod.

Notes:

1. `worker_replicas * gpus_per_worker` should match the intended global worker count for the job because `leap-finetune` schedules one Ray worker per GPU by default.
2. The head pod does not need GPUs unless you explicitly set `head_gpu_count`.
3. If you do not want the helper to create the cluster, skip the `kuberay:` section entirely and just set `ray.address` or `RAY_ADDRESS` to an existing KubeRay head service such as `ray://<cluster>-head-svc:10001` or the in-cluster `"auto"` path for submitted Ray jobs.
4. Concrete manifests for both patterns live in [examples/kuberay](/home/alay/leap-finetune-24b/examples/kuberay): a self-contained `RayJob`, a persistent `RayCluster`, and a launcher `Job` that attaches with `RAY_ADDRESS`.

### 3. (Optional) Experiment Tracking

Add `tracker` to your `training_config`:

```yaml
training_config:
  tracker: "trackio" # or "wandb"
```

#### Trackio

[Trackio](https://huggingface.co/blog/trackio) is a free experiment tracker that logs to a HuggingFace Space.

```yaml
training_config:
  tracker: "trackio"
  trackio_space_id: "username/my-dashboard" # auto-created if it doesn't exist
```

Requires a HF token (via `huggingface-cli login`). On Modal, the token is auto-injected — no extra setup needed. View your dashboard at `https://huggingface.co/spaces/<trackio_space_id>`.

#### Weights & Biases

[Weights & Biases](https://wandb.ai) is a popular experiment tracking platform.

```yaml
training_config:
  tracker: "wandb"
```

Set your API key locally with `export WANDB_API_KEY=your_key`. On Modal, add a secret:

```bash
modal secret create wandb-secret WANDB_API_KEY=your_key
```

Then add it to your Modal config:

```yaml
modal:
  secrets:
    - "wandb-secret"
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

### 🔧 Tool Calling Datasets

Tool calls use LFM bracket notation pre-baked in the assistant `content` field. Tool definitions go in the system prompt, and tool responses use `role: "tool"`.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "List of tools: [{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}},{\"type\":\"function\",\"function\":{\"name\":\"search_web\",\"description\":\"Search the web\",\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}}},{\"type\":\"function\",\"function\":{\"name\":\"send_email\",\"description\":\"Send an email\",\"parameters\":{\"type\":\"object\",\"properties\":{\"to\":{\"type\":\"string\"},\"body\":{\"type\":\"string\"}},\"required\":[\"to\",\"body\"]}}}]"
    },
    { "role": "user", "content": "What's the weather in Boston?" },
    {
      "role": "assistant",
      "content": "<|tool_call_start|>[get_weather(location=\"Boston\")]<|tool_call_end|>"
    },
    {
      "role": "tool",
      "content": "{\"temperature\": 72, \"condition\": \"sunny\"}"
    },
    { "role": "assistant", "content": "It's 72°F and sunny in Boston." }
  ]
}
```

- Tool calls must be pre-baked in `content` using `<|tool_call_start|>[func(args)]<|tool_call_end|>` bracket notation
- Structured `tool_calls` fields (OpenAI format) are auto-converted if present. If an assistant turn contains both prose and structured tool calls, LFM2.5/24B keeps prose before the tool-call marker to match the official template; legacy LFM2 keeps tool-call-first ordering for backward compatibility.
- Foreign formats (e.g. `<tool_call>` XML) are rejected with an actionable error
- Do not pre-wrap `role: "tool"` messages. Legacy LFM2 templates add `<|tool_response_start|>` / `<|tool_response_end|>` during tokenization; the LFM2.5/24B Shopify template keeps tool responses as bare ChatML `tool` turns.
- **LFM2 models** additionally expect `<|tool_list_start|>` / `<|tool_list_end|>` around tool definitions in the system prompt. Include these in your data if training an LFM2 model; omit them for LFM2.5. The pipeline warns on mismatches and auto-strips `<|tool_list_start|>` when training LFM2.5.
- Canonical tracked templates live in `job_configs/chat_templates/lfm2_tool_call_chat_template.jinja` for legacy LFM2 and `job_configs/chat_templates/lfm25_tool_call_chat_template.jinja` for LFM2.5/24B-style models.

## 🔄 Resuming Training

If a run is interrupted (GPU timeout, crash, SLURM preemption, etc.), you can resume from the last checkpoint with full optimizer state, LR schedule, and wandb continuity.

Add `resume_from_checkpoint` to your `training_config`:

```yaml
training_config:
  resume_from_checkpoint: "latest" # resumes from the most recent checkpoint
```

This finds the most recent run directory under `outputs/{project_name}/` and resumes from its latest checkpoint. To resume from a specific checkpoint instead:

```yaml
training_config:
  resume_from_checkpoint: "/path/to/outputs/my_project/run_name/checkpoint-step-8000"
```

**What gets restored:** model weights, optimizer states, LR scheduler position, training step counter, and RNG states. **In order to resume a run,** `save_only_model` **\*needs to be set to** `False`.

**Wandb continuity:** The wandb run ID is saved to `<run_dir>/.wandb_run_id` automatically. On resume, it restores the same wandb run. Fresh runs always get a new wandb run.

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
