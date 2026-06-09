from types import SimpleNamespace

import pytest
from datasets import Dataset

from leap_finetune.training.dpo import LFMDPOTrainer
from leap_finetune.training.utils.trainer_lifecycle import run_training_safely
from leap_finetune.training.utils.trainer_mixins import (
    validate_manual_sharded_training_args,
)


class FailingTrainer:
    def __init__(self, *, global_step=0, max_steps=-1, epoch=None, num_train_epochs=1):
        self.state = SimpleNamespace(global_step=global_step, epoch=epoch)
        self.args = SimpleNamespace(
            max_steps=max_steps, num_train_epochs=num_train_epochs
        )

    def train(self, **kwargs):
        raise RuntimeError("NCCL error during cleanup")


def test_run_training_safely_suppresses_cleanup_error_after_max_steps(caplog):
    trainer = FailingTrainer(global_step=10, max_steps=10)

    run_training_safely(trainer)

    assert "Training completed but hit distributed communication error" in caplog.text


def test_run_training_safely_raises_distributed_error_before_max_steps():
    trainer = FailingTrainer(global_step=9, max_steps=10)

    with pytest.raises(RuntimeError, match="NCCL error"):
        run_training_safely(trainer)


def test_run_training_safely_suppresses_cleanup_error_after_epoch_target(caplog):
    trainer = FailingTrainer(
        global_step=10, max_steps=-1, epoch=1.0, num_train_epochs=1
    )

    run_training_safely(trainer)

    assert "Training completed but hit distributed communication error" in caplog.text


def test_lfm_dpo_trainer_skips_prepare_dataset():
    dummy = Dataset.from_dict({"a": [1, 2, 3]})

    result = LFMDPOTrainer._prepare_dataset(None, dummy)

    assert result is dummy


def test_validate_manual_sharded_training_args_accepts_raw_checkpoint_format():
    validate_manual_sharded_training_args({}, checkpoint_format="both")


def test_validate_manual_sharded_training_args_rejects_raw_checkpoint_format():
    with pytest.raises(ValueError, match="manual_sharded_checkpoint_format"):
        validate_manual_sharded_training_args({}, checkpoint_format="invalid")
