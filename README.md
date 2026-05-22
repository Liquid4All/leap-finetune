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
<a href="#-setup">Setup</a> · <a href="#-quickstart">Quickstart</a> · <a href="#-expected-dataset-formats">Dataset Formats</a> · <a href="#grpo-group-relative-policy-optimization">GRPO</a> · <a href="#-tool-calling-datasets">Tool Calling</a> · <a href="#-resuming-training">Resuming Training</a> · <a href="#-evaluation-benchmarks">Benchmarks</a> · <a href="#-advanced-configuration">Advanced Config</a>
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

For CUDA / NVIDIA clusters, CUDA dependencies are included by default:

```bash
uv sync
```

For AMD / ROCm clusters, install the ROCm dependency group instead of the default CUDA group:

```bash
uv sync --no-group cuda --group rocm
```

The ROCm group is lockfile-managed and uses vLLM's ROCm wheel index for vLLM plus
its matching `torch`, `torchvision`, `torchaudio`, `flash-attn`, and `triton`
stack. The currently pinned vLLM ROCm wheels are Python 3.12 Linux wheels, so use
the repo's `.python-version` when creating AMD environments.

`flash-attn` is built against the selected Torch/CUDA ABI. If you previously
synced with a different Torch, CUDA, or vLLM version and see an error like
`flash_attn_2_cuda... undefined symbol`, clear the cached build and recreate
the environment:

```bash
uv cache clean flash-attn
rm -rf .venv
MAX_JOBS=1 uv sync
```

Run this on a machine with a CUDA toolkit and enough build memory available if
uv needs to rebuild `flash-attn` from source. `MAX_JOBS=1` keeps the fallback
source build from spawning too many CUDA compiler jobs at once.

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

That's it. In attached mode (`detach: false`), the CLI will:

1. Build the container image (~5 min on first run, cached after that)
2. Auto-create a `huggingface-secret` on Modal from your local HF token
3. Stream build and training logs to your terminal in real-time
4. Save checkpoints to a Modal Volume

**Retrieving checkpoints:**

```bash
modal volume ls leap-finetune                                        # list saved checkpoints
modal volume get leap-finetune <checkpoint-name> ./local-outputs     # download to local
```

**Detached mode:** Set `detach: true` in the modal config to submit and disconnect. The CLI prints the Modal app ID for that run, plus commands to monitor or stop it:

```bash
modal app logs ap-...
modal app stop ap-...
```

Detached runs are ephemeral Modal apps, so use the printed `ap-...` app ID rather than the `app_name` value from your config when viewing logs.

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

## Contents

- [Expected Dataset Formats](#-expected-dataset-formats)
- [GRPO](#grpo-group-relative-policy-optimization)
- [Tool Calling Datasets](#-tool-calling-datasets)
- [Resuming Training](#-resuming-training)
- [Evaluation Benchmarks](#-evaluation-benchmarks)
- [Advanced Configuration](#-advanced-configuration)

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

### GRPO and VLM GRPO

GRPO can reuse the SFT/VLM SFT `messages` format. The loader turns each
row into `prompt` and `solution` for online reward computation, and any
extra columns are forwarded to reward functions. See the
[GRPO section](#grpo-group-relative-policy-optimization) for the full
dataset, reward, and vLLM rollout contract.

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
- Structured `tool_calls` fields (OpenAI format) are auto-converted if present
- Foreign formats (e.g. `<tool_call>` XML) are rejected with an actionable error
- Do not include `<|tool_response_start|>` / `<|tool_response_end|>` markers in `role: "tool"` messages — the LFM2 chat template adds these automatically during tokenization
- **LFM2 models** additionally expect `<|tool_list_start|>` / `<|tool_list_end|>` around tool definitions in the system prompt. Include these in your data if training an LFM2 model; omit them for LFM2.5. The pipeline warns on mismatches and auto-strips `<|tool_list_start|>` when training LFM2.5.

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

**What gets restored:** model weights, optimizer states, LR scheduler position, training step counter, and RNG states. To resume a run, `save_only_model` must be set to `False`.

**Wandb continuity:** The wandb run ID is saved to `<run_dir>/.wandb_run_id` automatically. On resume, it restores the same wandb run. Fresh runs always get a new wandb run.

## GRPO (Group Relative Policy Optimization)

GRPO runs online RL with TRL v1's `GRPOTrainer`. Use
`training_type: "grpo"` for text models and `training_type: "vlm_grpo"`
for vision-language models. Both modes use the same YAML entrypoint as
SFT/DPO, the same Ray Train launcher, and vLLM rollouts by default.

**Dataset contract.** Text GRPO can reuse the SFT `messages` format: the
loader splits each row into `prompt` (non-assistant turns) and `solution`
(the last assistant message). Native `prompt` / `solution` columns also
work. VLM GRPO uses the same multimodal `messages` shape as VLM SFT;
`dataset.image_root` is prepended to relative image paths. Any extra
dataset columns are forwarded to reward functions as keyword arguments.

**Rewards.** The `rewards:` block resolves plain Python callables and
task recipes from [`rewards/`](./rewards/README.md). Shipped primitive
functions can be referenced by function name:

```yaml
rewards:
  funcs:
    - "accuracy_reward"
    - "length_reward"
  weights: [1.0, 0.1]
```

Task recipes bundle multiple reward functions and their default weights:

```yaml
rewards:
  recipe: "tasks/vlm_grounding/recipe.py::VLMGroundingIoURecipe"
```

If you combine `recipe:` and `funcs:`, the final reward order is recipe
rewards first, then individual funcs. A `weights:` override must match
that expanded order. Absolute paths and explicit
`./rewards/file.py::function_name` specs still work for custom rewards.

**Judge LLM reward.** Add `rewards.judge` when the reward signal should
come from an LLM grader. Without `base_url`, the driver starts a local
`trl vllm-serve` judge server before Ray initializes and exports the
endpoint to workers:

```yaml
rewards:
  judge:
    model: "LFM2-1.2B"
    weight: 1.0
    prompt_template: |
      Prompt:
      {prompt}

      Assistant response:
      {completion}

      Reference answer or rubric:
      {solution}

      Return only JSON: {"score": 0.0}

grpo_rollout:
  judge_gpus: 1
```

For an external judge, set `rewards.judge.base_url` and omit
`judge_gpus`. Judge scores are parsed from JSON or the first number in
the response and normalized from `min_score`/`max_score` to `[0, 1]`.

**vLLM rollout modes.** `DEFAULT_GRPO` and `DEFAULT_VLM_GRPO` set
`use_vllm: true` and `vllm_mode: "colocate"`.

- **Colocate** — vLLM runs inside each training worker and shares GPU
  memory. This is the default and works on single-node and multi-node
  jobs.
- **Server** — the driver starts `trl vllm-serve` before Ray initializes,
  then narrows `CUDA_VISIBLE_DEVICES` so Ray only sees training GPUs.
  Configure counts, not device ids:

```yaml
grpo_rollout:
  server_gpus: 1 # reserve 1 local GPU for vLLM; training gets the rest
  judge_gpus: 1 # optional: reserve 1 local GPU for rewards.judge
  # training_gpus: 3    # or set only this and vLLM gets the remaining GPUs
  tensor_parallel_size: 1
  dtype: "bfloat16"
  gpu_memory_utilization: 0.9

training_config:
  extends: "DEFAULT_GRPO"
  vllm_mode: "server"
  vllm_server_host: "auto"
  vllm_server_port: 8000
```

Local server partitioning is single-node only. For multi-node GRPO, use
colocate mode or point `vllm_server_base_url` at an externally managed
vLLM server without setting `server_gpus` / `training_gpus`.

**Example configs** — copy and edit instead of writing YAML from scratch:

- [`job_configs/grpo_example.yaml`](./job_configs/grpo_example.yaml) —
  text GRPO quickstart with the GSM8K recipe.
- [`job_configs/grpo_server_mode_example.yaml`](./job_configs/grpo_server_mode_example.yaml)
  — text GRPO with a local `trl vllm-serve` rollout server.
- [`job_configs/vlm_grpo_grounding_example.yaml`](./job_configs/vlm_grpo_grounding_example.yaml)
  — VLM GRPO with the visual-grounding recipe.

Launch the same way as SFT/DPO:

```bash
uv run leap-finetune job_configs/grpo_example.yaml
```

**Agentic environments (advanced).** For tasks where the environment
state evolves from agent actions (browsing, tool use, game simulators,
stateful multi-turn), `leap-finetune` also supports
[OpenEnv](https://github.com/meta-pytorch/OpenEnv) via an optional
`rl_env:` block. Install with `uv sync --extra rl-env` and see
[`src/leap_finetune/rl/environments/README.md`](./src/leap_finetune/rl/environments/README.md).
For anything scorable by a pure Python function, prefer the `rewards:`
path above — it is simpler and faster.

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

### Async Eval (vLLM)

By default, every `eval_steps` blocks training until benchmarks finish. For large generation suites this dominates wall-clock time. Add an `async_eval` block to run benchmarks **without blocking training**, using vLLM for the actual generation. Results are logged to wandb at the **same global-step axis** as today — dashboards look identical.

Three modes (default is `sync` = today's behavior):

| Mode | Pauses training? | GPUs reserved | Latency | Best for |
|---|---|---|---|---|
| `sync` | Yes | None | Immediate | Small/fast eval suites; default |
| `sidecar` | **No** | None (slurm-scheduled per cycle) | Slurm queue + eval time | Tight clusters; eval should be free of training cost |
| `reserved` | **No** | N throughout the run | ~30–60s respawn per cycle | Customers with spare GPUs who want predictable latency |

Both async modes serve generation through vLLM (TRL `vllm-serve`). Logprob benchmarks fall back to a sync HF model inside the runner if the chosen vLLM build doesn't support them.

```yaml
# Opt in by adding this block. See job_configs/sft_with_async_eval_example.yaml
async_eval:
  mode: sidecar          # sync (default) | sidecar | reserved
  vllm_gpus: 1
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9

  # mode=sidecar: short sbatch job per eval_steps
  sbatch:
    time: "00:30:00"
    # partition / account default to inheriting from the parent job

  # mode=reserved: long-running vllm-serve on dedicated GPUs (single-node only for v1)
  reserved:
    weight_reload: respawn
    server_port: 8100
```

Failures are isolated: if eval crashes or sbatch is rejected, training continues. After `failure.max_consecutive` consecutive failures the callback disables itself for the rest of the run. See [`job_configs/sft_with_async_eval_example.yaml`](./job_configs/sft_with_async_eval_example.yaml) for a full example.

### Post-Training Evaluation with lmms-eval

For comprehensive post-training evaluation on standard VLM benchmarks (MMMU, OCRBench, RefCOCO, POPE, etc.), install the optional `lmms-eval` extra:

```bash
uv sync --extra lmms-eval
```

This installs [lmms-eval](https://github.com/Liquid4All/lmms-eval) with built-in LFM2-VL model support.

**Evaluate a fine-tuned checkpoint:**

```bash
# Single GPU
python -m lmms_eval \
    --model lfm2_vl \
    --model_args pretrained=/path/to/checkpoint \
    --tasks mmmu_val,ocrbench,pope \
    --batch_size 1

# Multi-GPU
torchrun --nproc-per-node=4 -m lmms_eval \
    --model lfm2_vl \
    --model_args pretrained=/path/to/checkpoint \
    --tasks mmmu_val,ocrbench,pope \
    --batch_size 1
```

**For faster evaluation with vLLM backend (~8x speedup):**

```bash
uv sync --extra lmms-eval-vllm

python -m lmms_eval \
    --model lfm2_vl_vllm \
    --model_args pretrained=/path/to/checkpoint,tensor_parallel_size=1,gpu_memory_utilization=0.85 \
    --tasks mmmu_val,ocrbench,pope \
    --batch_size 64
```

**Updating lmms-eval to latest:**

```bash
uv lock --upgrade-package lmms-eval
uv sync --extra lmms-eval
```

> **Note:** Requires SSH access to the Liquid4All GitHub repos. The lmms-eval and vllm packages are sourced from private Liquid4All forks with LFM2 model support.

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
