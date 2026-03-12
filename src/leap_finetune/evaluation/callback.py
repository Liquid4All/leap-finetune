"""Generic benchmark evaluation callback for HuggingFace Trainer.

Model-agnostic — all inference and scoring logic lives in the Benchmark instances.
Each benchmark returns a ``BenchmarkResult`` with a dict of named metrics;
the callback all-reduces each metric across ranks and logs to wandb.


"""

import logging
import time

import torch
import torch.distributed as dist
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from leap_finetune.evaluation.base import Benchmark
from leap_finetune.utils.logging_utils import is_rank_zero

logger = logging.getLogger(__name__)


class BenchmarkEvalCallback(TrainerCallback):
    """Runs ``Benchmark.evaluate()`` at every eval step.

    Handles: sample sharding, all-reduce of arbitrary metrics, wandb logging,
    and model eval/train toggle.
    """

    def __init__(self, benchmarks: list[Benchmark]):
        super().__init__()
        self.benchmarks = benchmarks

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if model is None or not self.benchmarks:
            return

        rank, world_size = self._get_rank_and_world()
        unwrapped = self._unwrap_model(model)
        device = next(unwrapped.parameters()).device

        was_training = unwrapped.training
        unwrapped.eval()

        all_results = {}
        total_start = time.time()

        if rank == 0:
            logger.info(
                "\n%s\nBenchmark Evaluation (step %d)\n%s",
                "=" * 50,
                state.global_step,
                "=" * 50,
            )

        with torch.no_grad():
            for benchmark in self.benchmarks:
                samples = benchmark.get_samples()
                if not samples:
                    continue

                my_samples = samples[rank::world_size]

                start = time.time()
                result = benchmark.evaluate(unwrapped, my_samples, device)

                # All-reduce: pack [metric_values..., count] into a single tensor
                metric_names = sorted(result.metrics.keys())
                values = [result.metrics[k] for k in metric_names] + [
                    float(result.count)
                ]
                tensor = torch.tensor(values, device=device)
                if dist.is_initialized():
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

                total_count = int(tensor[-1].item())
                elapsed = time.time() - start

                for i, metric_name in enumerate(metric_names):
                    total_val = tensor[i].item()
                    avg = total_val / total_count if total_count > 0 else 0.0
                    all_results[f"benchmark/{benchmark.name}/{metric_name}"] = avg

                    if rank == 0:
                        logger.info(
                            "  %-20s %-12s %8.4f  (%d samples, %.1fs)",
                            benchmark.name,
                            metric_name,
                            avg,
                            total_count,
                            elapsed,
                        )

        total_elapsed = time.time() - total_start
        if rank == 0:
            logger.info(
                "%s\nTotal benchmark eval time: %.1fs\n%s",
                "=" * 50,
                total_elapsed,
                "=" * 50,
            )

        self._log_to_wandb(all_results)

        if was_training:
            unwrapped.train()

    @staticmethod
    def _get_rank_and_world() -> tuple[int, int]:
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def _log_to_wandb(results: dict):
        if not is_rank_zero():
            return
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(results, commit=False)
        except ImportError:
            pass
        except Exception as e:
            logger.warning("Failed to log to wandb: %s", e)
