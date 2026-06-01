import numpy as np
import pytest
from datasets import Dataset
from PIL import Image
from trl.trainer.dpo_trainer import DataCollatorForVisionPreference

from conftest import write_config
from leap_finetune.config.parser import parse_job_config
from leap_finetune.data_loading.validate_dataset_format import validate_vlm_dpo_format
from leap_finetune.training.vlm_dpo import PathLoadingVisionPreferenceCollator

pytestmark = pytest.mark.vlm


def _write_image(path):
    Image.new("RGB", (2, 2), color="red").save(path)
    return str(path)


def _valid_row(image_path):
    return {
        "prompt": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
        "chosen": [{"role": "assistant", "content": "good"}],
        "rejected": [{"role": "assistant", "content": "bad"}],
        "image": image_path,
    }


def test_parse_vlm_dpo_config(tmp_path):
    config = {
        "project_name": "vlm_dpo_test",
        "model_name": "LFM2-VL-1.6B",
        "training_type": "vlm_dpo",
        "dataset": {
            "path": "dummy",
            "type": "vlm_dpo",
            "test_size": None,
        },
        "training_config": {"extends": "DEFAULT_VLM_DPO", "eval_strategy": "no"},
    }

    job = parse_job_config(write_config(config, tmp_path))

    assert job.training_type == "vlm_dpo"
    assert job.dataset.dataset_type == "vlm_dpo"
    assert job.training_config.value["precompute_ref_log_probs"] is False
    assert job.training_config.value["eval_strategy"] == "no"


def test_validate_vlm_dpo_accepts_image_path(tmp_path):
    image_path = _write_image(tmp_path / "image.png")
    dataset = Dataset.from_list([_valid_row(image_path)])

    validated = validate_vlm_dpo_format(dataset)

    assert len(validated) == 1


def test_validate_vlm_dpo_rejects_missing_image():
    row = _valid_row("/tmp/missing.png")
    row.pop("image")
    dataset = Dataset.from_list([row])

    with pytest.raises(ValueError, match="VLM DPO needs"):
        validate_vlm_dpo_format(dataset)


def test_validate_vlm_dpo_rejects_identical_preferences(tmp_path):
    image_path = _write_image(tmp_path / "image.png")
    row = _valid_row(image_path)
    row["rejected"] = row["chosen"]
    dataset = Dataset.from_list([row])

    with pytest.raises(ValueError, match="chosen == rejected"):
        validate_vlm_dpo_format(dataset)


def test_validate_vlm_dpo_rejects_unloadable_image():
    dataset = Dataset.from_list([_valid_row("/tmp/does-not-exist.png")])

    with pytest.raises(ValueError, match="invalid VLM DPO"):
        validate_vlm_dpo_format(dataset)


def test_path_loading_vision_preference_collator_opens_and_closes_images(
    monkeypatch, tmp_path
):
    image_path = _write_image(tmp_path / "image.png")
    captured = {}

    def fake_torch_call(self, examples):
        del self
        captured["examples"] = examples
        assert isinstance(examples[0]["image"], Image.Image)
        assert isinstance(examples[0]["prompt"], list)
        return {"ok": True}

    monkeypatch.setattr(
        DataCollatorForVisionPreference,
        "torch_call",
        fake_torch_call,
    )
    collator = object.__new__(PathLoadingVisionPreferenceCollator)

    result = collator.torch_call(
        [
            {
                "prompt": np.array([{"role": "user", "content": "x"}], dtype=object),
                "chosen": [{"role": "assistant", "content": "good"}],
                "rejected": [{"role": "assistant", "content": "bad"}],
                "image": image_path,
            }
        ]
    )

    assert result == {"ok": True}
    with pytest.raises(ValueError):
        captured["examples"][0]["image"].load()
