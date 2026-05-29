from ray import train
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from leap_finetune.checkpointing.paths import (
    current_checkpoint_output_dir,
    rename_standard_checkpoint,
)


class LeapCheckpointCallback(TrainerCallback):
    """Integrates HuggingFace Trainer with Ray Train checkpointing and
    renames checkpoint dirs to descriptive names on every save.

    Renaming happens immediately in on_save (not on_train_end) so that
    checkpoints are properly named even if training crashes mid-run.

    Only rank 0 performs filesystem operations to avoid race conditions.

    Args:
        run_name_template: Template for checkpoint names, e.g.
            "LFM2-1.2B-sft-smoltalk-1000-lr2e005-w0p2-lora_a-20250217_143022"
    """

    def __init__(
        self,
        run_name_template: str | None = None,
        *,
        manual_sharded: bool = False,
    ) -> None:
        super().__init__()
        self.metrics: dict = {}
        self.loss_history: list[float] = []
        self.run_name_template = run_name_template
        self.manual_sharded = manual_sharded

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs:
            self.metrics.update(logs)
            if "loss" in logs:
                self.loss_history.append(logs["loss"])

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        checkpoint_path = current_checkpoint_output_dir(
            output_dir=args.output_dir,
            run_name_template=self.run_name_template,
            epoch=state.epoch,
            step=state.global_step,
            manual_sharded=self.manual_sharded,
        )
        if train.get_context().get_world_rank() == 0:
            print(f"Checkpoint saved: {checkpoint_path}")
            print(f"   Metrics: {self.metrics}")

            if not self.manual_sharded:
                rename_standard_checkpoint(
                    output_dir=args.output_dir,
                    run_name_template=self.run_name_template,
                    epoch=state.epoch,
                    step=state.global_step,
                    save_total_limit=args.save_total_limit,
                )

        # Include loss curve summary for test assertions
        report_metrics = self.metrics.copy()
        if self.loss_history:
            report_metrics["loss_history"] = self.loss_history.copy()

        # Report metrics only — HF Trainer already saved checkpoint to output_dir.
        # Passing checkpoint=None avoids Ray duplicating files into ray_logs/.
        train.report(metrics=report_metrics, checkpoint=None)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self.metrics:
            report_metrics = self.metrics.copy()
            if self.loss_history:
                report_metrics["loss_history"] = self.loss_history.copy()
            train.report(metrics=report_metrics, checkpoint=None)
