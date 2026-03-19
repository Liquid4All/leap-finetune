"""Generic benchmark evaluation callback for HuggingFace Trainer.

Model-agnostic — all inference and scoring logic lives in the Benchmark instances.
Each benchmark returns a ``BenchmarkResult`` with a dict of named metrics;
the callback all-reduces each metric across ranks and logs to wandb.


"""

import logging
import pathlib
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
    best-checkpoint tracking, and model eval/train toggle.
    """

    def __init__(
        self,
        benchmarks: list[Benchmark],
        best_metric_config: list[str] | dict[str, float] | None = None,
    ):
        super().__init__()
        self.benchmarks = benchmarks
        # Normalize to {name: weight} dict
        if isinstance(best_metric_config, list):
            self.best_metric_weights: dict[str, float] | None = {
                n: 1.0 for n in best_metric_config
            }
        elif isinstance(best_metric_config, dict):
            self.best_metric_weights = best_metric_config
        else:
            self.best_metric_weights = None
        self.best_avg_score: float = -1.0

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
        # Eval -> save -> best checkpoint
        self._pending_best_results = all_results

        if was_training:
            unwrapped.train()

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Update best symlink after the checkpoint has been written."""
        if hasattr(self, "_pending_best_results") and self._pending_best_results:
            self._update_best_checkpoint(self._pending_best_results, args, state)
            self._pending_best_results = None

    @staticmethod
    def _get_rank_and_world() -> tuple[int, int]:
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    def _update_best_checkpoint(
        self,
        results: dict[str, float],
        args: TrainingArguments,
        state: TrainerState,
    ) -> None:
        """Update the ``best`` symlink if the current eval is the best so far."""
        if not self.best_metric_weights or not is_rank_zero():
            return

        # Collect weighted scores for the requested benchmark names
        weighted_sum = 0.0
        total_weight = 0.0
        for name, weight in self.best_metric_weights.items():
            key = f"benchmark/{name}/score"
            if key in results:
                weighted_sum += results[key] * weight
                total_weight += weight

        if total_weight == 0:
            return

        avg_score = weighted_sum / total_weight

        if avg_score <= self.best_avg_score:
            return

        self.best_avg_score = avg_score

        # Find the most recent checkpoint directory for this step
        output_dir = pathlib.Path(args.output_dir)
        # Match renamed checkpoints (contain step number) or default HF naming
        best_ckpt = None
        for candidate in output_dir.iterdir():
            if not candidate.is_dir() or candidate.is_symlink():
                continue
            if candidate.name == f"checkpoint-{state.global_step}":
                best_ckpt = candidate
                break
            if f"s{state.global_step}-" in candidate.name:
                best_ckpt = candidate
                break

        if best_ckpt is None:
            return

        # Update 'best' symlink
        best_link = output_dir / "best"
        try:
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink(missing_ok=True)
            best_link.symlink_to(best_ckpt.name)
            logger.info(
                "New best checkpoint: %s (avg score: %.4f)",
                best_ckpt.name,
                avg_score,
            )
        except OSError:
            pass

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
