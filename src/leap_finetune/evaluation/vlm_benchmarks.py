"""VLM benchmark implementations for vision-language model evaluation.

Two strategies:
- VLMGenerationBenchmark: generate text and score against ground truth.
- VLMLogprobBenchmark: zero-shot MCQ via per-option logprob comparison.

Benchmark data uses the same HF messages schema as the training pipeline::

    Generation::
        {"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": "/path/to/img.jpg"},
                {"type": "text", "text": "Describe this image."}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "A dog on a beach."}
            ]}
        ]}

    Logprob MCQ::
        {"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": "/path/to/img.jpg"},
                {"type": "text", "text": "What animal is this?"}
            ]}
        ], "options": ["cat", "dog", "bird"], "answer_id": 1}

Column aliases (``conversation``, ``conversations``, etc.) are auto-renamed
to ``messages`` by the shared normalization layer in ``data_loaders.py``.
"""

import copy
import logging
from collections.abc import Callable

import numpy as np
import torch

from leap_finetune.data_loaders.image_loader import load_image
from leap_finetune.evaluation.base import Benchmark, BenchmarkResult
from leap_finetune.evaluation.data_loaders import load_benchmark_samples
from leap_finetune.evaluation.metrics import compute_metric

logger = logging.getLogger(__name__)


class VLMGenerationBenchmark(Benchmark):
    """Generate text with a VLM and score against ground truth.

    The last conversation turn (assistant) is treated as ground truth;
    prior turns form the prompt.
    """

    def __init__(
        self,
        name: str,
        path: str,
        processor,
        metric: str | Callable = "short_answer",
        max_new_tokens: int = 128,
        match_mode: str = "contains",
        limit: int | None = None,
        format: str | None = None,
        image_root: str | None = None,
        **metric_kwargs,
    ):
        super().__init__(name)
        self.path = path
        self.processor = processor
        self.metric = metric
        self.max_new_tokens = max_new_tokens
        self.match_mode = match_mode
        self.limit = limit
        self.format = format
        self.image_root = image_root
        self.metric_kwargs = metric_kwargs

    def load_samples(self) -> list[dict]:
        return load_benchmark_samples(
            self.path, self.limit, self.format, self.image_root
        )

    def evaluate(self, model, samples: list[dict], device) -> BenchmarkResult:
        total_score = 0.0
        count = 0
        for sample in samples:
            try:
                total_score += self._score_sample(model, sample, device)
            except Exception:
                logger.warning(
                    "[%s] Failed on sample %s",
                    self.name,
                    sample.get("id", count),
                    exc_info=True,
                )
            count += 1
        return BenchmarkResult(metrics={"score": total_score}, count=count)

    def _score_sample(self, model, sample: dict, device) -> float:
        messages = sample["messages"]
        ground_truth = _extract_text(messages[-1]["content"])

        prompt, loaded = _prepare_messages(messages[:-1])
        try:
            inputs = self.processor.apply_chat_template(
                [prompt],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )

            prompt_len = inputs["input_ids"].shape[1]
            prediction = self.processor.tokenizer.decode(
                output_ids[0, prompt_len:], skip_special_tokens=True
            ).strip()

            if callable(self.metric):
                return self.metric(prediction, ground_truth, **self.metric_kwargs)
            return compute_metric(
                self.metric,
                prediction,
                ground_truth,
                match_mode=self.match_mode,
                **self.metric_kwargs,
            )
        finally:
            _close_images(loaded)


class VLMLogprobBenchmark(Benchmark):
    """Zero-shot MCQ classification — picks the highest-logprob answer option.

    Runs one forward pass per option, sums the log-probabilities of the answer
    tokens, and selects the option with the highest total.
    """

    def __init__(
        self,
        name: str,
        path: str,
        processor,
        limit: int | None = None,
        format: str | None = None,
        image_root: str | None = None,
    ):
        super().__init__(name)
        self.path = path
        self.processor = processor
        self.limit = limit
        self.format = format
        self.image_root = image_root

    def load_samples(self) -> list[dict]:
        return load_benchmark_samples(
            self.path, self.limit, self.format, self.image_root
        )

    def evaluate(self, model, samples: list[dict], device) -> BenchmarkResult:
        total_score = 0.0
        count = 0
        for sample in samples:
            try:
                total_score += self._score_sample(model, sample, device)
            except Exception:
                logger.warning(
                    "[%s] Failed on sample %s",
                    self.name,
                    sample.get("id", count),
                    exc_info=True,
                )
            count += 1
        return BenchmarkResult(metrics={"score": total_score}, count=count)

    def _score_sample(self, model, sample: dict, device) -> float:
        options = sample["options"]
        answer_id = int(sample["answer_id"])

        prompt, loaded = _prepare_messages(sample["messages"])
        try:
            prompt_inputs = self.processor.apply_chat_template(
                [prompt],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]

            option_scores = []
            for option in options:
                full_conv = prompt + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": option.strip()}],
                    }
                ]
                full_inputs = self.processor.apply_chat_template(
                    [full_conv],
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                full_inputs = {
                    k: v.to(device) for k, v in full_inputs.items() if v is not None
                }

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**full_inputs).logits

                log_probs = logits[0].log_softmax(dim=-1)
                input_ids = full_inputs["input_ids"][0]
                n_tokens = len(input_ids) - prompt_len
                total_logprob = sum(
                    log_probs[i - 1, input_ids[i].item()].item()
                    for i in range(prompt_len, len(input_ids))
                )
                option_scores.append(total_logprob / n_tokens if n_tokens > 0 else total_logprob)

            return 1.0 if int(np.argmax(option_scores)) == answer_id else 0.0
        finally:
            _close_images(loaded)


# -- Helpers --


def _prepare_messages(messages: list[dict]) -> tuple[list[dict], list]:
    """Deep-copy messages and load images from string paths into PIL objects.

    Same pattern as the training collate_fn in tokenize_data.py — replaces
    path strings with PIL images so the processor can encode them.
    """
    messages = copy.deepcopy(messages)
    loaded = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image" and isinstance(item.get("image"), str):
                img = load_image(item["image"])
                item["image"] = img
                loaded.append(img)
    return messages, loaded


def _extract_text(content) -> str:
    """Extract plain text from message content (string or structured list)."""
    if isinstance(content, str):
        return content
    return " ".join(item["text"] for item in content if item.get("type") == "text")


def _close_images(images: list) -> None:
    for img in images:
        if hasattr(img, "close"):
            img.close()
