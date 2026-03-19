import pytest
from transformers import TrainingArguments

from leap_finetune.utils.checkpoint_callback import LeapCheckpointCallback

pytestmark = pytest.mark.configs


class TestCheckpointCallback:
    def test_create_without_template(self):
        cb = LeapCheckpointCallback()
        assert cb.run_name_template is None
        assert cb.metrics == {}

    def test_create_with_template(self):
        cb = LeapCheckpointCallback(run_name_template="test-run-20250101")
        assert cb.run_name_template == "test-run-20250101"

    def test_on_log_accumulates_metrics(self, tmp_path):
        cb = LeapCheckpointCallback()
        args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
        state = type("State", (), {"epoch": 1, "global_step": 10})()
        control = type("Control", (), {})()

        cb.on_log(args, state, control, logs={"loss": 1.5, "lr": 1e-4})
        assert cb.metrics == {"loss": 1.5, "lr": 1e-4}

        # Later log overwrites earlier values
        cb.on_log(args, state, control, logs={"loss": 0.8})
        assert cb.metrics["loss"] == 0.8
        assert cb.metrics["lr"] == 1e-4  # preserved

    def test_on_log_ignores_none(self, tmp_path):
        cb = LeapCheckpointCallback()
        args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
        state = type("State", (), {})()
        control = type("Control", (), {})()

        cb.on_log(args, state, control, logs=None)
        assert cb.metrics == {}

    def test_rename_checkpoint(self, tmp_path):
        cb = LeapCheckpointCallback(
            run_name_template="LFM2-sft-smoltalk-20250101_120000"
        )
        # Create a fake checkpoint dir
        checkpoint_dir = tmp_path / "checkpoint-100"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.safetensors").touch()

        args = TrainingArguments(
            output_dir=str(tmp_path), report_to="none", save_strategy="no"
        )
        state = type("State", (), {"epoch": 1.0, "global_step": 100})()

        cb._rename_checkpoint(args, state)

        # Original should be gone
        assert not checkpoint_dir.exists()
        # Renamed dir should exist with epoch/step pattern
        renamed = list(tmp_path.glob("LFM2-sft-smoltalk-e1s100-*"))
        assert len(renamed) == 1
        assert (renamed[0] / "model.safetensors").exists()
        # Latest symlink should point to it
        latest = tmp_path / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == renamed[0].resolve()

    def test_rotate_checkpoints(self, tmp_path):
        cb = LeapCheckpointCallback()
        # Create 4 checkpoint dirs
        for step in [10, 20, 30, 40]:
            d = tmp_path / f"model-e1s{step}-20250101"
            d.mkdir()
            (d / "model.safetensors").touch()

        cb._rotate_checkpoints(tmp_path, limit=2)

        remaining = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
        # Should keep the 2 newest (highest step): s30 and s40
        assert len(remaining) == 2
        assert "model-e1s30-20250101" in remaining
        assert "model-e1s40-20250101" in remaining
