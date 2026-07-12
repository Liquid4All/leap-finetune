from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from leap_finetune.checkpointing.paths import current_checkpoint_output_dir
from leap_finetune.state.store import (
    record_checkpoint,
    record_eval_result,
    update_run_fields,
    update_run_progress,
)


def _is_rank_zero() -> bool:
    try:
        from leap_finetune.training.utils.logging import is_rank_zero

        return is_rank_zero()
    except Exception:
        return True


class LFTStateCallback(TrainerCallback):
    """Mirror compact Trainer progress into ``.lft/state.json``.

    External telemetry keeps owning full dashboards and artifact logging. This
    callback only stores the small facts agents need for quick run inspection.
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        run_name_template: str | None = None,
        manual_sharded: bool = False,
    ) -> None:
        super().__init__()
        self.run_id = run_id or os.environ.get("LFT_RUN_ID")
        self.run_name_template = run_name_template
        self.manual_sharded = manual_sharded

    def _enabled(self) -> bool:
        return bool(self.run_id) and _is_rank_zero()

    @staticmethod
    def _log_refs(args: TrainingArguments) -> dict[str, Any]:
        output_dir = (
            Path(args.output_dir) if getattr(args, "output_dir", None) else None
        )
        if output_dir is None:
            return {}
        async_eval_dir = output_dir / "_async_eval"
        return {
            "ray": str(output_dir / "ray_logs"),
            "async_eval": str(async_eval_dir),
            "async_eval_logs": str(async_eval_dir / "logs"),
            "reserved_vllm_server": str(async_eval_dir / "vllm_server" / "server.log"),
        }

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not self._enabled():
            return
        update_run_progress(
            self.run_id,
            phase="training",
            status="running",
            step=state.global_step,
            epoch=state.epoch,
            max_steps=getattr(state, "max_steps", None)
            or getattr(args, "max_steps", None),
            log_refs=self._log_refs(args),
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if not self._enabled():
            return
        update_run_progress(
            self.run_id,
            phase="training",
            status="running",
            step=state.global_step,
            epoch=state.epoch,
            max_steps=getattr(state, "max_steps", None)
            or getattr(args, "max_steps", None),
            log=logs,
            source="trainer",
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not self._enabled():
            return
        metrics = kwargs.get("metrics")
        if isinstance(metrics, dict) and metrics:
            eval_metrics = {
                key: value
                for key, value in metrics.items()
                if not str(key).startswith("benchmark/")
            }
            if not eval_metrics:
                return
            record_eval_result(
                self.run_id,
                step=state.global_step,
                metrics=eval_metrics,
                status="completed",
                source="trainer_eval",
            )
        else:
            update_run_progress(
                self.run_id,
                phase="evaluating",
                status="running",
                step=state.global_step,
                epoch=state.epoch,
            )

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not self._enabled():
            return
        checkpoint_path = current_checkpoint_output_dir(
            output_dir=args.output_dir,
            run_name_template=self.run_name_template,
            epoch=state.epoch,
            step=state.global_step,
            manual_sharded=self.manual_sharded,
        )
        record_checkpoint(
            self.run_id,
            path=checkpoint_path,
            step=state.global_step,
            epoch=state.epoch,
            source="trainer",
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not self._enabled():
            return
        update_run_fields(
            self.run_id,
            status="completed",
            phase="completed",
        )
        update_run_progress(
            self.run_id,
            phase="completed",
            status="completed",
            step=state.global_step,
            epoch=state.epoch,
            max_steps=getattr(state, "max_steps", None)
            or getattr(args, "max_steps", None),
        )


def add_lft_state_callback(
    trainer,
    *,
    run_name_template: str | None = None,
    manual_sharded: bool = False,
):
    if os.environ.get("LFT_RUN_ID"):
        trainer.add_callback(
            LFTStateCallback(
                run_name_template=run_name_template,
                manual_sharded=manual_sharded,
            )
        )
    return trainer
