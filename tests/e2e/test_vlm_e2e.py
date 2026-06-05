import pytest

from conftest import (
    assert_checkpoints_exist,
    assert_training_result,
    requires_gpu,
    run_e2e_training,
)

pytestmark = pytest.mark.vlm

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


# === VLM SFT with LoRA ===


class TestVLMLoRA:
    @requires_gpu
    def test_training_completes_and_learns(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_vlm_lora.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result)

    @requires_gpu
    def test_vlm_custom_optimizer_structure(self):
        """Verify LFMVLMTrainer creates per-component LR param groups."""
        import torch
        from transformers import TrainingArguments

        from leap_finetune.checkpointing.model_loading import load_vlm_model
        from leap_finetune.training.default_configs.vlm_sft_configs import (
            DEFAULT_LR_MULTIPLIERS,
        )
        from leap_finetune.training.vlm_sft import LFMVLMTrainer

        model, processor = load_vlm_model("LFM2-VL-1.6B")
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
    def test_training_completes_learns_and_checkpoints(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_vlm_full.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result)

        assert_checkpoints_exist(e2e_output_dir)


# === VLM SFT with string assistant content + multi-image ===


def _write_string_content_vlm_dataset(tmp_path):
    """Tiny VLM SFT dataset with string assistant content + a multi-image row.

    messages are JSON strings (mixed list/string content isn't parquet-safe).
    Returns the config yaml path.
    """
    import json

    import pandas as pd
    import yaml
    from PIL import Image

    imgdir = tmp_path / "images"
    imgdir.mkdir(parents=True, exist_ok=True)
    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
    paths = []
    for i, c in enumerate(colors):
        p = imgdir / f"img{i}.png"
        Image.new("RGB", (48, 48), c).save(p)
        paths.append(str(p))

    rows = []
    for i in range(40):
        if i % 7 == 0:  # multi-image rows
            content = [
                {"type": "image", "image": paths[i % 4]},
                {"type": "image", "image": paths[(i + 1) % 4]},
                {"type": "text", "text": "Compare the two images."},
            ]
            answer = "The two images differ in color."
        else:
            content = [
                {"type": "image", "image": paths[i % 4]},
                {"type": "text", "text": "What color is this?"},
            ]
            answer = "It is a colored square."
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": answer},  # STRING content
        ]
        rows.append({"messages": json.dumps(messages)})

    ds_path = tmp_path / "data.parquet"
    pd.DataFrame(rows).to_parquet(ds_path)

    config = {
        "project_name": "e2e_vlm_string",
        "model_name": "LFM2-VL-1.6B",
        "training_type": "vlm_sft",
        "dataset": {"path": str(ds_path), "type": "vlm_sft", "test_size": 0.2},
        "training_config": {
            "extends": "DEFAULT_VLM_SFT",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "learning_rate": 1e-5,
            "gradient_checkpointing": True,
            "save_strategy": "epoch",
            "eval_strategy": "epoch",
            "logging_steps": 2,
        },
        "peft_config": {"use_peft": True, "extends": "DEFAULT_VLM_LORA"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    return str(cfg_path)


class TestVLMStringContent:
    @requires_gpu
    def test_string_assistant_and_multi_image_train(self, e2e_output_dir, tmp_path):
        """Regression: VLM datasets with plain-string assistant content (and
        multi-image turns) must train, not be silently dropped at the row filter."""
        config_path = _write_string_content_vlm_dataset(tmp_path)
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result)
        assert_checkpoints_exist(e2e_output_dir)
