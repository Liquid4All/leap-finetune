"""Benchmark evaluation callback for VLM training.

Runs generation-based (and logprob-based) evaluation on JSONL benchmark datasets
at every eval step, scoring predictions against ground truth and logging results.
"""

import json
import logging
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from leap_finetune.data_loaders.image_loader import load_image
from leap_finetune.evaluation.metrics import compute_metric
from leap_finetune.utils.logging_utils import is_rank_zero

logger = logging.getLogger(__name__)

GENERATION_METRICS = {"grounding_iou", "short_answer", "mcq_gen"}
LOGPROB_METRICS = {"logprob_zero_shot"}


class BenchmarkEvalCallback(TrainerCallback):
    """Runs benchmark evaluation at every eval step during VLM training.

    Loads JSONL benchmark datasets, generates predictions or computes logprobs,
    scores against ground truth, and logs per-benchmark accuracy.
    Samples are sharded across ranks for distributed evaluation.

    Args:
        processor: VLM processor for tokenization and image processing.
        benchmark_configs: List of benchmark dicts from YAML config.
        default_max_new_tokens: Default max generation length.
    """

    def __init__(
        self,
        processor,
        benchmark_configs: list[dict],
        default_max_new_tokens: int = 128,
    ):
        super().__init__()
        self.processor = processor
        self.benchmark_configs = benchmark_configs
        self.default_max_new_tokens = default_max_new_tokens
        self._cache: dict[str, list[dict]] | None = None

    def _load_benchmarks(self) -> dict[str, list[dict]]:
        if self._cache is not None:
            return self._cache

        self._cache = {}
        for bench in self.benchmark_configs:
            name = bench["name"]
            path = bench["path"]
            limit = bench.get("limit")

            samples = []
            with open(path) as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    samples.append(json.loads(line))

            self._cache[name] = samples
            if is_rank_zero():
                logger.info(f"Loaded benchmark '{name}': {len(samples)} samples")

        return self._cache

    def _get_rank_and_world(self) -> tuple[int, int]:
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _unwrap_model(self, model):
        if hasattr(model, "module"):
            return model.module
        return model

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if model is None or not self.benchmark_configs:
            return

        rank, world_size = self._get_rank_and_world()
        unwrapped = self._unwrap_model(model)
        benchmarks = self._load_benchmarks()
        device = next(unwrapped.parameters()).device

        was_training = unwrapped.training
        unwrapped.eval()

        all_results = {}
        total_start = time.time()

        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Benchmark Evaluation (step {state.global_step})")
            print(f"{'='*50}")

        with torch.no_grad():
            for bench_config in self.benchmark_configs:
                name = bench_config["name"]
                metric_type = bench_config["metric"]
                max_new_tokens = bench_config.get("max_new_tokens", self.default_max_new_tokens)
                match_mode = bench_config.get("match_mode", "contains")

                samples = benchmarks.get(name, [])
                if not samples:
                    continue

                # Shard samples across ranks
                my_samples = samples[rank::world_size]

                start = time.time()
                my_total_score = 0.0
                my_count = 0

                for sample in my_samples:
                    try:
                        if metric_type in LOGPROB_METRICS:
                            score = self._evaluate_logprob_sample(
                                unwrapped, sample, device,
                            )
                        else:
                            score, _ = self._evaluate_generation_sample(
                                unwrapped, sample, metric_type,
                                max_new_tokens, match_mode,
                            )
                        my_total_score += score
                        my_count += 1
                    except Exception as e:
                        logger.warning(
                            f"Error in {name} sample {sample.get('id', '?')}: {e}\n"
                            f"{traceback.format_exc()}"
                        )
                        my_count += 1  # count as 0 score

                # All-reduce scores across ranks
                score_tensor = torch.tensor(
                    [my_total_score, float(my_count)], device=device
                )
                if dist.is_initialized():
                    dist.all_reduce(score_tensor, op=dist.ReduceOp.SUM)

                total_score = score_tensor[0].item()
                total_count = int(score_tensor[1].item())
                accuracy = total_score / total_count if total_count > 0 else 0.0
                elapsed = time.time() - start

                all_results[f"benchmark/{name}/accuracy"] = accuracy

                if rank == 0:
                    print(f"  {name:<20s} {accuracy*100:6.2f}%  ({total_count} samples, {elapsed:.1f}s)")

        total_elapsed = time.time() - total_start
        if rank == 0:
            print(f"{'='*50}")
            print(f"Total benchmark eval time: {total_elapsed:.1f}s")
            print(f"{'='*50}\n")

        self._log_to_wandb(all_results, state.global_step)

        if was_training:
            unwrapped.train()

    def _evaluate_generation_sample(
        self,
        model,
        sample: dict,
        metric_type: str,
        max_new_tokens: int,
        match_mode: str,
    ) -> tuple[float, dict]:
        conversation = sample["conversation"]
        image_paths = sample.get("images", [])

        prompt_messages = conversation[:-1]
        ground_truth = conversation[-1]["content"]

        loaded_images = [load_image(p) for p in image_paths]

        try:
            formatted = self._format_for_generation(prompt_messages, loaded_images)
            device = next(model.parameters()).device

            inputs = self.processor.apply_chat_template(
                [formatted],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0, prompt_len:]
            prediction = self.processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            score = compute_metric(
                metric_type, prediction, ground_truth, match_mode=match_mode
            )

            return score, {}
        finally:
            for img in loaded_images:
                if hasattr(img, "close"):
                    img.close()

    def _evaluate_logprob_sample(
        self,
        model,
        sample: dict,
        device: torch.device,
    ) -> float:
        conversation = sample["conversation"]
        image_paths = sample.get("images", [])
        options = sample["options"]
        answer_id = int(sample["answer_id"])

        prompt_messages = conversation[:-1]
        loaded_images = [load_image(p) for p in image_paths]

        try:
            formatted_prompt = self._format_for_generation(prompt_messages, loaded_images)

            # Tokenize prompt-only (with generation prompt) to get prompt length
            prompt_inputs = self.processor.apply_chat_template(
                [formatted_prompt],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]

            class_scores = []
            for option in options:
                # Build full conversation: prompt + assistant answer with option text
                full_conv = formatted_prompt + [
                    {"role": "assistant", "content": [{"type": "text", "text": option.strip()}]}
                ]

                # Let the processor handle all tokenization and image processing
                full_inputs = self.processor.apply_chat_template(
                    [full_conv],
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                full_inputs = {k: v.to(device) for k, v in full_inputs.items() if v is not None}

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(**full_inputs)

                logits = outputs.logits[0]  # (seq_len, vocab_size)
                log_probs = logits.log_softmax(dim=-1)

                # Sum log probs of tokens after the prompt position
                input_ids = full_inputs["input_ids"][0]
                total_logprob = 0.0
                for i in range(prompt_len, len(input_ids)):
                    tok_id = input_ids[i].item()
                    # position i-1 predicts token at position i
                    total_logprob += log_probs[i - 1, tok_id].item()

                class_scores.append(total_logprob)

            predicted_id = int(np.argmax(class_scores))
            return 1.0 if predicted_id == answer_id else 0.0
        finally:
            for img in loaded_images:
                if hasattr(img, "close"):
                    img.close()

    def _format_for_generation(
        self, prompt_messages: list[dict], images: list
    ) -> list[dict]:
        """Convert JSONL conversation format to processor chat format.

        JSONL: {"role": "user", "content": "<image>Question..."}
        Processor: {"role": "user", "content": [{"type": "image", "image": <PIL>}, {"type": "text", "text": "..."}]}
        """
        formatted = []
        image_idx = 0

        for msg in prompt_messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user" and "<image>" in content:
                parts = []
                text_parts = content.split("<image>")

                for i, text_part in enumerate(text_parts):
                    if i > 0 and image_idx < len(images):
                        parts.append({"type": "image", "image": images[image_idx]})
                        image_idx += 1
                    if text_part.strip():
                        parts.append({"type": "text", "text": text_part.strip()})

                formatted.append({"role": role, "content": parts})
            else:
                formatted.append(
                    {"role": role, "content": [{"type": "text", "text": content}]}
                )

        return formatted

    def _log_to_wandb(self, results: dict, step: int):
        if not is_rank_zero():
            return
        try:
            import wandb

            if wandb.run is not None:
                # Use commit=False to attach benchmark metrics to the current
                # trainer step.
                wandb.log(results, commit=False)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")
