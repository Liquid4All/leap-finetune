from types import SimpleNamespace

import pytest

from leap_finetune.utils.trainer_mixins import run_training_safely


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
