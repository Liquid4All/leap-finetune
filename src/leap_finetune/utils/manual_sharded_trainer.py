import logging

from leap_finetune.utils.model_utils import (
    save_manual_sharded_model_checkpoint,
    save_manual_sharded_trainer_checkpoint,
)

logger = logging.getLogger(__name__)


def validate_manual_sharded_training_args(config_kwargs: dict) -> None:
    """Reject Trainer args that conflict with the manual-sharded runtime."""
    if config_kwargs.get("gradient_checkpointing"):
        raise ValueError(
            "gradient_checkpointing=True is not supported in manual-sharded EP/FSDP2 "
            "runs. This path already applies activation checkpointing in the FSDP2 "
            "wrapper; remove gradient_checkpointing from training_config."
        )


class ManualShardedTrainerMixin:
    """Shared Trainer overrides for EP and non-EP manual-sharded modes."""

    manual_sharded: bool
    ep_config: dict | None

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        if getattr(self, "manual_sharded", False):

            def _manual_prepare_model(
                model, device_placement=None, evaluation_mode=False
            ):
                del device_placement, evaluation_mode
                self.accelerator._models.append(model)
                return model

            self.accelerator.prepare_model = _manual_prepare_model

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        if not getattr(self, "manual_sharded", False):
            return super().save_model(output_dir, _internal_call)

        output_dir = output_dir or self.args.output_dir
        mode_name = "EP" if self.ep_config is not None else "FSDP2"
        logger.info("%s trainer save_model start output_dir=%s", mode_name, output_dir)
        save_manual_sharded_model_checkpoint(
            model=self.model,
            accelerator=self.accelerator,
            output_dir=output_dir,
            processing_class=self.processing_class,
            data_collator=self.data_collator,
            training_args=self.args,
            ep_group=self.ep_config["ep_group"] if self.ep_config is not None else None,
        )
        logger.info("%s trainer save_model end output_dir=%s", mode_name, output_dir)

    def _save_checkpoint(self, model, trial) -> None:
        if not getattr(self, "manual_sharded", False):
            return super()._save_checkpoint(model, trial)

        step = getattr(getattr(self, "state", None), "global_step", None)
        output_dir = getattr(getattr(self, "args", None), "output_dir", None)
        mode_name = "EP" if self.ep_config is not None else "FSDP2"
        logger.info(
            "%s trainer _save_checkpoint start step=%s output_dir=%s",
            mode_name,
            step,
            output_dir,
        )
        save_manual_sharded_trainer_checkpoint(
            trainer=self,
            model=model,
            trial=trial,
            ep_group=self.ep_config["ep_group"] if self.ep_config is not None else None,
        )
        logger.info("%s trainer _save_checkpoint end step=%s", mode_name, step)
