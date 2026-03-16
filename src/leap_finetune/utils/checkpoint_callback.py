import pathlib
import re
import shutil

from ray import train
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState


_CLOUD_PREFIXES = ("s3://", "gs://", "az://", "abfs://", "abfss://", "hdfs://")


def _is_local_path(path: str) -> bool:
    return not path.startswith(_CLOUD_PREFIXES)


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

    def __init__(self, run_name_template: str | None = None) -> None:
        super().__init__()
        self.metrics: dict = {}
        self.loss_history: list[float] = []
        self.run_name_template = run_name_template

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
        if train.get_context().get_world_rank() == 0:
            checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"
            print(f"Checkpoint saved: {checkpoint_path}")
            print(f"   Metrics: {self.metrics}")

            if self.run_name_template and _is_local_path(args.output_dir):
                self._rename_checkpoint(args, state)

        # Include loss curve summary for test assertions
        report_metrics = self.metrics.copy()
        if self.loss_history:
            report_metrics["loss_history"] = self.loss_history.copy()

        # Report metrics only — HF Trainer already saved checkpoint to output_dir.
        # Passing checkpoint=None avoids Ray duplicating files into ray_logs/.
        train.report(metrics=report_metrics, checkpoint=None)
        self.metrics.clear()

    def _rename_checkpoint(self, args: TrainingArguments, state: TrainerState) -> None:
        output_path = pathlib.Path(args.output_dir)
        source = output_path / f"checkpoint-{state.global_step}"
        if not source.exists():
            return

        epoch = int(state.epoch) if state.epoch else 0
        step = state.global_step

        # Split template: everything before last "-" is base, last part is timestamp
        if "-" in self.run_name_template:
            base_part, time_part = self.run_name_template.rsplit("-", 1)
        else:
            base_part = self.run_name_template
            time_part = ""

        # Include step to guarantee uniqueness (multiple saves per epoch)
        new_name = f"{base_part}-e{epoch}s{step}"
        if time_part:
            new_name += f"-{time_part}"

        dest = output_path / new_name

        if dest.exists():
            return

        try:
            source.rename(dest)
        except OSError:
            return

        # Update 'latest' symlink
        latest_link = output_path / "latest"
        try:
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink(missing_ok=True)
            latest_link.symlink_to(new_name)
        except OSError:
            pass

        # Rotate old checkpoints if save_total_limit is set
        if args.save_total_limit is not None and args.save_total_limit > 0:
            self._rotate_checkpoints(output_path, args.save_total_limit)

    def _rotate_checkpoints(self, output_dir: pathlib.Path, limit: int) -> None:
        checkpoints = []
        for path in output_dir.iterdir():
            if (
                path.is_dir()
                and not path.name.startswith(".")
                and not path.is_symlink()
            ):
                # Match -e{epoch}s{step}- pattern (new) or -e{epoch}- (legacy)
                match = re.search(r"-e(\d+)s(\d+)-", path.name)
                if match:
                    step = int(match.group(2))
                    checkpoints.append((step, path))
                    continue
                match = re.search(r"-e(\d+)-", path.name)
                if match:
                    epoch_num = int(match.group(1))
                    checkpoints.append((epoch_num, path))

        checkpoints.sort(key=lambda x: x[0])

        if len(checkpoints) > limit:
            for _, path in checkpoints[: len(checkpoints) - limit]:
                shutil.rmtree(path)
