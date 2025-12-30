from typing import Callable

from ray import train
from ray.train import Checkpoint
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState


class LeapCheckpointCallback(TrainerCallback):
    """
    Callback that integrates HuggingFace Trainer with Ray Train checkpointing.

    With DeepSpeed ZeRO Stage 2, model parameters are fully replicated across
    workers (only optimizer states are sharded), so we save from rank 0 only.

    Args:
        on_checkpoint_saved: Optional hook called after checkpoint is saved.
                            Receives (checkpoint_path: str, metrics: dict)
    """

    def __init__(
        self, on_checkpoint_saved: Callable[[str, dict], None] | None = None
    ) -> None:
        super().__init__()
        self.metrics: dict = {}
        self.on_checkpoint_saved = on_checkpoint_saved

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        """Accumulate metrics from logging steps."""
        if logs:
            self.metrics.update(logs)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        checkpoint = None
        checkpoint_path = None

        if train.get_context().get_world_rank() == 0:
            # Get the latest checkpoint directory from HF Trainer
            checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"
            # Reference the checkpoint path directly (no copy)
            checkpoint = Checkpoint(path=checkpoint_path)

            # Log checkpoint save
            print(f"🔔 Checkpoint saved: {checkpoint_path}")
            print(f"   Metrics: {self.metrics}")

            # Call post-save hook if provided
            if self.on_checkpoint_saved and checkpoint_path:
                self.on_checkpoint_saved(checkpoint_path, self.metrics.copy())

        # Report to Ray Train with metrics and checkpoint
        train.report(metrics=self.metrics.copy(), checkpoint=checkpoint)

        # Clear metrics buffer after reporting
        self.metrics = {}
