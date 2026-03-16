import pathlib

import pytest

from conftest import assert_training_result, requires_gpu, run_e2e_training

pytestmark = pytest.mark.vlm

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# === VLM SFT with LoRA ===


class TestVLMLoRA:
    @requires_gpu
    def test_training_completes_and_learns(self, tmp_path):
        config_path = str(FIXTURES / "e2e_vlm_lora.yaml")
        result = run_e2e_training(config_path, tmp_path)
        assert_training_result(result)

    @requires_gpu
    def test_vlm_custom_optimizer_structure(self):
        """Verify LFMVLMTrainer creates per-component LR param groups."""
        import torch
        from transformers import TrainingArguments

        from leap_finetune.training_loops.vlm_sft_run import LFMVLMTrainer
        from leap_finetune.training_configs.vlm_sft_config import DEFAULT_LR_MULTIPLIERS
        from leap_finetune.utils.load_models import load_vlm_model

        model, processor = load_vlm_model("LFM2-1.2B")
        args = TrainingArguments(
            output_dir="/tmp/test_vlm_opt",
            learning_rate=1e-4,
            per_device_train_batch_size=1,
            report_to="none",
            max_steps=1,
        )
        trainer = LFMVLMTrainer(
            lr_multipliers=DEFAULT_LR_MULTIPLIERS,
            model=model,
            processing_class=processor,
            args=args,
        )
        optimizer = trainer.create_optimizer()

        assert isinstance(optimizer, torch.optim.AdamW)

        # Should have param groups for each prefix + possibly ungrouped
        groups = optimizer.param_groups
        assert len(groups) >= 2, f"Expected multiple param groups, got {len(groups)}"

        # Find the vision_tower group (should have lr = base_lr * 0.1)
        base_lr = args.learning_rate
        vision_group = None
        for g in groups:
            if abs(g["lr"] - base_lr * 0.1) < 1e-10:
                vision_group = g
                break
        assert vision_group is not None, (
            f"No param group with vision tower LR ({base_lr * 0.1}). "
            f"Groups have LRs: {[g['lr'] for g in groups]}"
        )


# === VLM SFT full fine-tune ===


class TestVLMFull:
    @requires_gpu
    def test_training_completes_and_learns(self, tmp_path):
        config_path = str(FIXTURES / "e2e_vlm_full.yaml")
        result = run_e2e_training(config_path, tmp_path)
        assert_training_result(result)

    @requires_gpu
    def test_checkpoint_exists(self, tmp_path):
        config_path = str(FIXTURES / "e2e_vlm_full.yaml")
        run_e2e_training(config_path, tmp_path)
        checkpoint_dirs = list(tmp_path.rglob("checkpoint-*"))
        assert len(checkpoint_dirs) > 0, "No checkpoint directories found"
