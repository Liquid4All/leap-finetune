"""Factory: YAML benchmark config → LLM Benchmark instances.

Reads the ``benchmarks:`` section from a job config and creates the right
benchmark class based on the ``metric`` field.

Available metrics for LLM benchmarks:
    - ``short_answer``      — substring match (generation)
    - ``mcq_gen``           — MCQ letter extraction (generation)
    - ``logprob_zero_shot`` — per-option logprob comparison (no generation)

Example YAML::

    benchmarks:
      max_new_tokens: 128
      benchmarks:
        - name: "reading_comprehension"
          path: "/data/reading_eval.jsonl"
          metric: "short_answer"

        - name: "phd_mcq"
          path: "/data/phd_logprob.jsonl"
          metric: "logprob_zero_shot"
"""

import logging

from leap_finetune.evaluation.base import Benchmark
from leap_finetune.evaluation.llm_benchmarks import (
    LLMGenerationBenchmark,
    LLMLogprobBenchmark,
)

logger = logging.getLogger(__name__)

GENERATION_METRICS = {"short_answer", "mcq_gen"}
LOGPROB_METRICS = {"logprob_zero_shot"}

# Keys consumed by the factory (not forwarded to benchmark constructors)
_FACTORY_KEYS = {"name", "path", "type"}


def create_llm_benchmarks_from_config(
    benchmark_configs: dict,
    tokenizer,
) -> list[Benchmark]:
    """Convert the ``benchmarks:`` YAML section into LLM Benchmark instances."""
    benchmarks_list = benchmark_configs.get("benchmarks", [])
    default_max_new_tokens = benchmark_configs.get("max_new_tokens", 128)

    result: list[Benchmark] = []
    for bench in benchmarks_list:
        name = bench["name"]
        path = bench["path"]
        metric = bench.get("metric", "")

        kwargs = {k: v for k, v in bench.items() if k not in _FACTORY_KEYS}
        kwargs.setdefault("max_new_tokens", default_max_new_tokens)

        if metric in LOGPROB_METRICS:
            result.append(
                LLMLogprobBenchmark(
                    name=name,
                    path=path,
                    tokenizer=tokenizer,
                    limit=bench.get("limit"),
                    format=bench.get("format"),
                )
            )
        elif metric in GENERATION_METRICS:
            result.append(
                LLMGenerationBenchmark(
                    name=name,
                    path=path,
                    tokenizer=tokenizer,
                    **kwargs,
                )
            )
        else:
            logger.warning(
                "Unknown metric %r for benchmark %r. Available: %s",
                metric,
                name,
                sorted(GENERATION_METRICS | LOGPROB_METRICS),
            )

    return result
