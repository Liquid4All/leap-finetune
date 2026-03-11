"""Factory: YAML benchmark config → Benchmark instances.

Reads the ``benchmarks:`` section from a job config and creates the right
benchmark class based on the ``metric`` field.

Available metrics for VLM benchmarks:
    - ``short_answer``      — substring match (generation)
    - ``grounding_iou``     — bounding box IoU (generation)
    - ``mcq_gen``           — MCQ letter extraction (generation)
    - ``logprob_zero_shot`` — per-option logprob comparison (no generation)

Example YAML::

    benchmarks:
      max_new_tokens: 128
      benchmarks:
        - name: "mmmu_val"
          path: "/data/mmmu_val.jsonl"
          metric: "short_answer"
          max_new_tokens: 50

        - name: "imagenette"
          path: "/data/imagenette_eval.jsonl"
          metric: "logprob_zero_shot"
"""

import logging

from leap_finetune.evaluation.base import Benchmark
from leap_finetune.evaluation.vlm_benchmarks import (
    VLMGenerationBenchmark,
    VLMLogprobBenchmark,
)

logger = logging.getLogger(__name__)

GENERATION_METRICS = {"grounding_iou", "short_answer", "mcq_gen"}
LOGPROB_METRICS = {"logprob_zero_shot"}

# Keys consumed by the factory (not forwarded to benchmark constructors)
_FACTORY_KEYS = {"name", "path", "type"}


def create_vlm_benchmarks_from_config(
    benchmark_configs: dict,
    processor,
) -> list[Benchmark]:
    """Convert the ``benchmarks:`` YAML section into Benchmark instances."""
    benchmarks_list = benchmark_configs.get("benchmarks", [])
    default_max_new_tokens = benchmark_configs.get("max_new_tokens", 128)
    default_image_root = benchmark_configs.get("image_root")

    result: list[Benchmark] = []
    for bench in benchmarks_list:
        name = bench["name"]
        path = bench["path"]
        metric = bench.get("metric", "")

        # Remaining YAML keys are forwarded as kwargs to the benchmark constructor
        kwargs = {k: v for k, v in bench.items() if k not in _FACTORY_KEYS}
        kwargs.setdefault("max_new_tokens", default_max_new_tokens)
        kwargs.setdefault("image_root", default_image_root)

        if metric in LOGPROB_METRICS:
            result.append(
                VLMLogprobBenchmark(
                    name=name, path=path, processor=processor,
                    limit=bench.get("limit"), format=bench.get("format"),
                    image_root=bench.get("image_root", default_image_root),
                )
            )
        elif metric in GENERATION_METRICS:
            result.append(
                VLMGenerationBenchmark(
                    name=name, path=path, processor=processor, **kwargs,
                )
            )
        else:
            logger.warning(
                "Unknown metric %r for benchmark %r. "
                "Available: %s",
                metric, name, sorted(GENERATION_METRICS | LOGPROB_METRICS),
            )

    return result
