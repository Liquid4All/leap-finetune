import pathlib

import yaml
from peft import LoraConfig

from leap_finetune.utils.config_parser import parse_job_config


def _write_config(config: dict, tmp_path: pathlib.Path) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


BASE_DATASET = {
    "path": "HuggingFaceTB/smoltalk",
    "type": "sft",
    "limit": 10,
    "test_size": 0.2,
    "subset": "all",
}


class TestPeftExtendsOverride:
    def test_override_r_value(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"extends": "DEFAULT_LORA", "use_peft": True, "r": 32},
        }
        job = parse_job_config(_write_config(config, tmp_path))
        peft_val = job.peft_config.value
        assert isinstance(peft_val, LoraConfig)
        assert peft_val.r == 32

    def test_override_alpha_preserves_r(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {
                "extends": "DEFAULT_LORA",
                "use_peft": True,
                "lora_alpha": 64,
            },
        }
        job = parse_job_config(_write_config(config, tmp_path))
        peft_val = job.peft_config.value
        assert peft_val.lora_alpha == 64
        assert peft_val.r == 8  # Preserved from DEFAULT_LORA

    def test_use_peft_false_disables(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(_write_config(config, tmp_path))
        assert job.peft_config is None

    def test_use_peft_true_without_extends_uses_default(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": True},
        }
        job = parse_job_config(_write_config(config, tmp_path))
        from leap_finetune.training_configs import PeftConfig

        assert job.peft_config is PeftConfig.DEFAULT_LORA

    def test_no_peft_section_gives_none(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
        }
        job = parse_job_config(_write_config(config, tmp_path))
        assert job.peft_config is None

    def test_extends_vlm_lora_with_dropout_override(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "vlm_sft",
            "dataset": {
                "path": "alay2shah/example-vlm-sft-dataset",
                "type": "vlm_sft",
                "limit": 10,
                "test_size": 0.2,
            },
            "training_config": {"extends": "DEFAULT_VLM_SFT"},
            "peft_config": {
                "extends": "DEFAULT_VLM_LORA",
                "use_peft": True,
                "lora_dropout": 0.3,
            },
        }
        job = parse_job_config(_write_config(config, tmp_path))
        peft_val = job.peft_config.value
        assert peft_val.lora_dropout == 0.3
        assert peft_val.r == 8  # From DEFAULT_VLM_LORA

    def test_extends_high_r_lora(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"extends": "HIGH_R_LORA", "use_peft": True},
        }
        job = parse_job_config(_write_config(config, tmp_path))
        peft_val = job.peft_config.value
        assert peft_val.r == 16
        assert peft_val.lora_alpha == 32

    def test_peft_value_accessible_via_dot_value(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"extends": "DEFAULT_LORA", "use_peft": True, "r": 4},
        }
        job = parse_job_config(_write_config(config, tmp_path))
        # Must be accessible via .value (mimics enum interface)
        assert hasattr(job.peft_config, "value")
        assert job.peft_config.value.r == 4


class TestTrainingConfigOverrides:
    def test_learning_rate_string_coercion(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_DATASET,
            "training_config": {
                "extends": "DEFAULT_SFT",
                "learning_rate": "1e-4",
            },
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(_write_config(config, tmp_path))
        lr = job.training_config.value["learning_rate"]
        assert isinstance(lr, float)
        assert lr == 1e-4

    def test_backward_compat_no_extends(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "dpo",
            "dataset": {
                "path": "mlabonne/orpo-dpo-mix-40k",
                "type": "dpo",
                "limit": 10,
                "test_size": 0.2,
                "subset": "default",
            },
            "training_config": {
                "num_train_epochs": 5,
                "learning_rate": 2e-6,
            },
        }
        job = parse_job_config(_write_config(config, tmp_path))
        config_val = job.training_config.value
        # Should fallback to DEFAULT_DPO
        assert config_val["training_type"] == "dpo"
        assert config_val["beta"] == 0.1  # From DEFAULT_DPO
        assert config_val["num_train_epochs"] == 5  # Overridden
        assert config_val["learning_rate"] == 2e-6  # Overridden

    def test_base_keyword_works_like_extends(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_DATASET,
            "training_config": {
                "base": "DEFAULT_SFT",
                "num_train_epochs": 10,
            },
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(_write_config(config, tmp_path))
        assert job.training_config.value["num_train_epochs"] == 10
        assert "deepspeed" in job.training_config.value
