import os
import pathlib
import tempfile

import pytest
import yaml
from peft import LoraConfig

from leap_finetune.utils.checkpoint_callback import LeapCheckpointCallback
from leap_finetune.utils.config_parser import (
    generate_run_name,
    parse_job_config,
    resolve_config_path,
)
from leap_finetune.utils.constants import LEAP_FINETUNE_DIR

from conftest import BASE_SFT_DATASET, write_config

pytestmark = pytest.mark.configs


BASE_DPO_DATASET = {
    "path": "mlabonne/orpo-dpo-mix-40k",
    "type": "dpo",
    "limit": 10,
    "test_size": 0.2,
    "subset": "default",
}

BASE_VLM_DATASET = {
    "path": "alay2shah/example-vlm-sft-dataset",
    "type": "vlm_sft",
    "limit": 10,
    "test_size": 0.2,
}


# === Fixtures ===


@pytest.fixture
def job_configs_dir():
    return LEAP_FINETUNE_DIR / "job_configs"


@pytest.fixture
def sft_config_path(job_configs_dir):
    return str(job_configs_dir / "sft_example.yaml")


@pytest.fixture
def dpo_config_path(job_configs_dir):
    return str(job_configs_dir / "dpo_example.yaml")


@pytest.fixture
def vlm_config_path(job_configs_dir):
    return str(job_configs_dir / "vlm_sft_example.yaml")


@pytest.fixture
def moe_sft_config_path(job_configs_dir):
    return str(job_configs_dir / "moe_sft_example.yaml")


@pytest.fixture
def moe_dpo_config_path(job_configs_dir):
    return str(job_configs_dir / "moe_dpo_example.yaml")


@pytest.fixture
def slurm_config_path(job_configs_dir):
    return str(job_configs_dir / "sft_example_with_slurm.yaml")


# === Config path resolution ===


class TestResolveConfigPath:
    def test_absolute_path(self, sft_config_path):
        result = resolve_config_path(sft_config_path)
        assert result.exists()
        assert result.name == "sft_example.yaml"

    def test_repo_job_configs(self):
        result = resolve_config_path("sft_example.yaml")
        assert result.exists()

    def test_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            resolve_config_path("nonexistent_config.yaml")


# === Run name generation ===


class TestGenerateRunName:
    def test_basic_format(self):
        name = generate_run_name(
            model_name="LFM2-1.2B",
            training_type="sft",
            dataset_path="HuggingFaceTB/smoltalk",
            dataset_limit=1000,
            learning_rate=2e-5,
            warmup_ratio=0.2,
            use_peft=True,
            lora_type="a",
        )
        assert "LFM2-1.2B" in name
        assert "sft" in name
        assert "smoltalk" in name
        assert "1000" in name
        assert "lora_a" in name

    def test_no_peft(self):
        name = generate_run_name(
            model_name="LFM2-1.2B",
            training_type="dpo",
            dataset_path="some/dataset",
            dataset_limit=None,
            learning_rate=None,
            warmup_ratio=None,
            use_peft=False,
        )
        assert "no_lora" in name
        assert "all" in name
        assert "lr_def" in name
        assert "w_def" in name

    def test_long_dataset_name_truncated(self):
        name = generate_run_name(
            model_name="m",
            training_type="sft",
            dataset_path="very_long_dataset_name_here",
            dataset_limit=None,
            learning_rate=None,
            warmup_ratio=None,
            use_peft=False,
        )
        parts = name.split("-")
        assert len(parts[2]) <= 10


# === YAML parsing ===


class TestParseJobConfig:
    def test_parse_sft_example(self, sft_config_path):
        job = parse_job_config(sft_config_path)
        assert job.training_type == "sft"
        assert job.model_name == "LFM2-1.2B"
        assert job.job_name == "my_sft_project"

    def test_parse_dpo_example(self, dpo_config_path):
        job = parse_job_config(dpo_config_path)
        assert job.training_type == "dpo"
        assert job.job_name == "my_dpo_project"

    def test_parse_vlm_example(self, vlm_config_path):
        job = parse_job_config(vlm_config_path)
        assert job.training_type == "vlm_sft"
        assert job.job_name == "my_vlm_project"

    def test_parse_moe_sft_example(self, moe_sft_config_path):
        job = parse_job_config(moe_sft_config_path)
        assert job.training_type == "sft"
        assert job.model_name == "LFM2-8B-A1B"

    def test_parse_moe_dpo_example(self, moe_dpo_config_path):
        job = parse_job_config(moe_dpo_config_path)
        assert job.training_type == "dpo"
        assert job.model_name == "LFM2-8B-A1B"


# === Config extends/inheritance ===


class TestExtendsResolution:
    def test_extends_default_sft(self, sft_config_path):
        job = parse_job_config(sft_config_path)
        config = job.training_config.value
        assert config["num_train_epochs"] == 3
        assert config["per_device_train_batch_size"] == 2
        assert config["learning_rate"] == 2e-5
        assert config["training_type"] == "sft"
        assert "deepspeed" in config

    def test_extends_moe_sft(self, moe_sft_config_path):
        job = parse_job_config(moe_sft_config_path)
        config = job.training_config.value
        assert config["num_train_epochs"] == 2
        assert config["per_device_train_batch_size"] == 2

    def test_peft_extends_default_lora(self, sft_config_path):
        job = parse_job_config(sft_config_path)
        peft_value = job.peft_config.value
        assert peft_value is not None
        assert hasattr(peft_value, "r")
        assert peft_value.r == 8

    def test_peft_extends_moe_lora(self, moe_sft_config_path):
        job = parse_job_config(moe_sft_config_path)
        peft_value = job.peft_config.value
        assert peft_value is not None
        assert peft_value.target_modules == "all-linear"


# === PEFT overrides ===


class TestPeftOverrides:
    def test_override_r_value(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"extends": "DEFAULT_LORA", "use_peft": True, "r": 32},
        }
        job = parse_job_config(write_config(config, tmp_path))
        peft_val = job.peft_config.value
        assert isinstance(peft_val, LoraConfig)
        assert peft_val.r == 32

    def test_override_alpha_preserves_r(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {
                "extends": "DEFAULT_LORA",
                "use_peft": True,
                "lora_alpha": 64,
            },
        }
        job = parse_job_config(write_config(config, tmp_path))
        peft_val = job.peft_config.value
        assert peft_val.lora_alpha == 64
        assert peft_val.r == 8

    def test_use_peft_false_disables(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        assert job.peft_config is None

    def test_use_peft_true_without_extends_uses_default(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": True},
        }
        job = parse_job_config(write_config(config, tmp_path))
        from leap_finetune.training_configs import PeftConfig

        assert job.peft_config is PeftConfig.DEFAULT_LORA

    def test_no_peft_section_gives_none(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
        }
        job = parse_job_config(write_config(config, tmp_path))
        assert job.peft_config is None

    def test_extends_vlm_lora_with_dropout_override(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "vlm_sft",
            "dataset": BASE_VLM_DATASET,
            "training_config": {"extends": "DEFAULT_VLM_SFT"},
            "peft_config": {
                "extends": "DEFAULT_VLM_LORA",
                "use_peft": True,
                "lora_dropout": 0.3,
            },
        }
        job = parse_job_config(write_config(config, tmp_path))
        peft_val = job.peft_config.value
        assert peft_val.lora_dropout == 0.3
        assert peft_val.r == 8

    def test_extends_high_r_lora(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"extends": "HIGH_R_LORA", "use_peft": True},
        }
        job = parse_job_config(write_config(config, tmp_path))
        peft_val = job.peft_config.value
        assert peft_val.r == 16
        assert peft_val.lora_alpha == 32

    def test_peft_value_accessible_via_dot_value(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"extends": "DEFAULT_LORA", "use_peft": True, "r": 4},
        }
        job = parse_job_config(write_config(config, tmp_path))
        assert hasattr(job.peft_config, "value")
        assert job.peft_config.value.r == 4


# === Training config overrides ===


class TestTrainingConfigOverrides:
    def test_learning_rate_string_coercion(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT", "learning_rate": "1e-4"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        lr = job.training_config.value["learning_rate"]
        assert isinstance(lr, float)
        assert lr == 1e-4

    def test_backward_compat_no_extends(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "dpo",
            "dataset": BASE_DPO_DATASET,
            "training_config": {"num_train_epochs": 5, "learning_rate": 2e-6},
        }
        job = parse_job_config(write_config(config, tmp_path))
        config_val = job.training_config.value
        assert config_val["training_type"] == "dpo"
        assert config_val["beta"] == 0.1
        assert config_val["num_train_epochs"] == 5
        assert config_val["learning_rate"] == 2e-6

    def test_base_keyword_works_like_extends(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"base": "DEFAULT_SFT", "num_train_epochs": 10},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        assert job.training_config.value["num_train_epochs"] == 10
        assert "deepspeed" in job.training_config.value


# === All example configs ===


class TestAllExampleConfigs:
    @pytest.fixture
    def job_configs_dir(self):
        return LEAP_FINETUNE_DIR / "job_configs"

    @pytest.mark.parametrize(
        "config_name",
        [
            "sft_example.yaml",
            "dpo_example.yaml",
            "vlm_sft_example.yaml",
            "moe_sft_example.yaml",
            "moe_dpo_example.yaml",
            "sft_example_with_slurm.yaml",
            "sft_with_lora_example.yaml",
        ],
    )
    def test_parse_and_to_dict(self, job_configs_dir, config_name):
        config_path = str(job_configs_dir / config_name)
        job = parse_job_config(config_path)

        d = job.to_dict()
        assert isinstance(d["training_config"], dict)
        assert "output_dir" in d["training_config"]
        assert "leap_run_name_template" in d["training_config"]
        assert d["model_name"]
        assert d["training_type"] in ("sft", "dpo", "vlm_sft")


# === Output dir and env overrides ===


class TestOutputAndEnv:
    def test_output_dir_created(self, tmp_path):
        config = {
            "project_name": "test_output_creation",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        output_dir = pathlib.Path(job.training_config.value["output_dir"])
        assert output_dir.exists()

    def test_output_dir_env_override(self, tmp_path, monkeypatch):
        env_dir = str(tmp_path / "env_output")
        pathlib.Path(env_dir).mkdir()
        monkeypatch.setenv("OUTPUT_DIR", env_dir)
        config = {
            "project_name": "test_env_override",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        assert job.training_config.value["output_dir"] == env_dir

    def test_dataset_path_env_is_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATASET_PATH", "custom/override/dataset")
        config = {
            "project_name": "test_ds_override",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        path = write_config(config, tmp_path)
        config_path = resolve_config_path(path)
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        ds_config = config_dict.get("dataset", {})
        dataset_path_env = os.getenv("DATASET_PATH")
        final_path = dataset_path_env if dataset_path_env else ds_config.get("path")
        assert final_path == "custom/override/dataset"


# === Run name in config ===


class TestRunNameInConfig:
    def test_run_name_has_model_info(self, tmp_path):
        config = {
            "project_name": "test_rn",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT", "learning_rate": 2e-5},
            "peft_config": {"extends": "DEFAULT_LORA", "use_peft": True},
        }
        job = parse_job_config(write_config(config, tmp_path))
        run_name = job.training_config.value["leap_run_name_template"]
        assert "LFM2" in run_name
        assert "sft" in run_name
        assert "lora_a" in run_name

    def test_run_name_no_peft(self, tmp_path):
        config = {
            "project_name": "test_rn",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        run_name = job.training_config.value["leap_run_name_template"]
        assert "no_lora" in run_name


# === Checkpoint callback ===


class TestCheckpointCallback:
    def test_create_without_template(self):
        cb = LeapCheckpointCallback()
        assert cb.run_name_template is None
        assert cb.metrics == {}

    def test_create_with_template(self):
        cb = LeapCheckpointCallback(run_name_template="test-run-20250101")
        assert cb.run_name_template == "test-run-20250101"


# === SLURM generation ===


class TestSlurmGeneration:
    def test_slurm_config_parsed(self, slurm_config_path):
        with open(slurm_config_path) as f:
            config = yaml.safe_load(f)
        assert "slurm" in config
        assert config["slurm"]["job_name"] == "sft_training"
        assert config["slurm"]["gpus_per_task"] == 4

    def test_generate_slurm_script(self, slurm_config_path):
        from leap_finetune.utils.slurm_generator import generate_slurm_script

        config_path = pathlib.Path(slurm_config_path)
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = pathlib.Path(tmpdir)
            script_path = generate_slurm_script(
                config_path, config_dict, output_dir, auto_submit=False
            )
            assert script_path.exists()
            content = script_path.read_text()
            assert "#!/bin/bash" in content
            assert "#SBATCH --job-name=sft_training" in content
            assert "#SBATCH --gpus-per-task=4" in content
            assert "LEAP_FINETUNE_FROM_SLURM=1" in content
            assert "leap-finetune" in content


# === Invalid configs ===


class TestInvalidConfigs:
    def test_unknown_extends_raises(self, tmp_path):
        config = {
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "NONEXISTENT_CONFIG"},
        }
        with pytest.raises(ValueError, match="Unknown base config"):
            parse_job_config(write_config(config, tmp_path))

    def test_unknown_peft_extends_raises(self, tmp_path):
        config = {
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"extends": "NONEXISTENT_LORA", "use_peft": True},
        }
        with pytest.raises(ValueError, match="Unknown base PEFT config"):
            parse_job_config(write_config(config, tmp_path))

    def test_unknown_training_type_raises(self, tmp_path):
        config = {
            "model_name": "LFM2-1.2B",
            "training_type": "invalid_type",
            "dataset": BASE_SFT_DATASET,
        }
        with pytest.raises(ValueError, match="Unknown training type"):
            parse_job_config(write_config(config, tmp_path))


# === to_dict ===


class TestJobConfigToDict:
    def test_to_dict_has_required_keys(self, sft_config_path):
        job = parse_job_config(sft_config_path)
        d = job.to_dict()
        assert "model_name" in d
        assert "job_name" in d
        assert "training_type" in d
        assert "training_config" in d
        assert "dataset" in d
        assert "peft_config" in d

    def test_to_dict_training_config_is_dict(self, sft_config_path):
        job = parse_job_config(sft_config_path)
        d = job.to_dict()
        assert isinstance(d["training_config"], dict)

    def test_run_name_template_in_training_config(self, sft_config_path):
        job = parse_job_config(sft_config_path)
        d = job.to_dict()
        assert "leap_run_name_template" in d["training_config"]

    def test_to_dict_peft_is_lora_config(self, sft_config_path):
        job = parse_job_config(sft_config_path)
        d = job.to_dict()
        assert isinstance(d["peft_config"], LoraConfig)

    def test_to_dict_no_peft(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        d = job.to_dict()
        assert d["peft_config"] is None


# === Project name fallback ===


class TestProjectNameFallback:
    def test_project_name_used(self, sft_config_path):
        job = parse_job_config(sft_config_path)
        assert job.job_name == "my_sft_project"

    def test_job_name_fallback(self, tmp_path):
        config = {
            "job_name": "test_job",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        assert job.job_name == "test_job"

    def test_default_fallback(self, tmp_path):
        config = {
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        assert job.job_name == "default_job"


# ============================================================================
# NEW: Config pipeline integrity tests
# ============================================================================


class TestConfigPipelineIntegrity:
    def test_sft_lr_survives_pipeline(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT", "learning_rate": 2e-5},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        d = job.to_dict()
        # Simulate filtering as done in sft_run.py
        from leap_finetune.training_configs.sft_configs import SFT_EXCLUDED_KEYS

        excluded = SFT_EXCLUDED_KEYS | {"leap_run_name_template"}
        filtered = {k: v for k, v in d["training_config"].items() if k not in excluded}
        assert filtered["learning_rate"] == 2e-5

    def test_dpo_beta_survives_pipeline(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "dpo",
            "dataset": BASE_DPO_DATASET,
            "training_config": {"extends": "DEFAULT_DPO", "beta": 0.3},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        d = job.to_dict()
        excluded = {"training_type", "wandb_logging", "leap_run_name_template"}
        filtered = {k: v for k, v in d["training_config"].items() if k not in excluded}
        assert filtered["beta"] == 0.3

    def test_vlm_lr_survives_pipeline(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "vlm_sft",
            "dataset": BASE_VLM_DATASET,
            "training_config": {"extends": "DEFAULT_VLM_SFT", "learning_rate": 1e-5},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        d = job.to_dict()
        from leap_finetune.training_configs.vlm_sft_config import VLM_SFT_EXCLUDED_KEYS

        excluded = VLM_SFT_EXCLUDED_KEYS | {"leap_run_name_template"}
        filtered = {k: v for k, v in d["training_config"].items() if k not in excluded}
        assert filtered["learning_rate"] == 1e-5

    def test_sft_filtering_removes_only_excluded_keys(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT", "learning_rate": 2e-5},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        d = job.to_dict()
        from leap_finetune.training_configs.sft_configs import SFT_EXCLUDED_KEYS

        excluded = SFT_EXCLUDED_KEYS | {"leap_run_name_template"}
        filtered = {k: v for k, v in d["training_config"].items() if k not in excluded}
        for key in SFT_EXCLUDED_KEYS:
            assert key not in filtered
        assert "leap_run_name_template" not in filtered
        assert "learning_rate" in filtered
        assert "output_dir" in filtered
        assert "deepspeed" in filtered

    def test_dpo_filtering_removes_only_excluded_keys(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "dpo",
            "dataset": BASE_DPO_DATASET,
            "training_config": {"extends": "DEFAULT_DPO"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        d = job.to_dict()
        excluded = {"training_type", "wandb_logging", "leap_run_name_template"}
        filtered = {k: v for k, v in d["training_config"].items() if k not in excluded}
        assert "training_type" not in filtered
        assert "learning_rate" in filtered
        assert "beta" in filtered
        assert "deepspeed" in filtered

    def test_vlm_filtering_removes_only_excluded_keys(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "vlm_sft",
            "dataset": BASE_VLM_DATASET,
            "training_config": {"extends": "DEFAULT_VLM_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        d = job.to_dict()
        from leap_finetune.training_configs.vlm_sft_config import VLM_SFT_EXCLUDED_KEYS

        excluded = VLM_SFT_EXCLUDED_KEYS | {"leap_run_name_template"}
        filtered = {k: v for k, v in d["training_config"].items() if k not in excluded}
        for key in VLM_SFT_EXCLUDED_KEYS:
            assert key not in filtered
        assert "learning_rate" in filtered
        assert "deepspeed" in filtered

    def test_fsdp_replaces_deepspeed_for_moe_no_peft(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-8B-A1B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "MOE_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        d = job.to_dict()
        from leap_finetune.training_configs.distributed_configs import MOE_FSDP_CONFIG
        from leap_finetune.training_configs.sft_configs import SFT_EXCLUDED_KEYS
        from leap_finetune.utils.model_utils import is_moe_model_from_name

        is_moe = is_moe_model_from_name(d["model_name"])
        use_fsdp = is_moe and d["peft_config"] is None
        assert use_fsdp

        # Simulate the filtering in sft_run.py
        excluded = SFT_EXCLUDED_KEYS | {"leap_run_name_template", "deepspeed"}
        filtered = {k: v for k, v in d["training_config"].items() if k not in excluded}
        assert "deepspeed" not in filtered

        # FSDP config would be injected
        config_kwargs = {**filtered}
        config_kwargs["fsdp"] = MOE_FSDP_CONFIG["fsdp"]
        config_kwargs["fsdp_config"] = MOE_FSDP_CONFIG["fsdp_config"]
        assert "fsdp" in config_kwargs
        assert "shard_grad_op" in config_kwargs["fsdp"]

    def test_full_finetune_no_peft_config(self, tmp_path):
        config = {
            "project_name": "test",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        d = job.to_dict()
        assert d["peft_config"] is None


# ============================================================================
# NEW: DeepSpeed config structure tests
# ============================================================================


class TestDeepSpeedConfigStructure:
    def test_sft_deepspeed_has_optimizer(self):
        from leap_finetune.training_configs.sft_configs import DEEPSPEED_CONFIG

        assert "optimizer" in DEEPSPEED_CONFIG
        assert DEEPSPEED_CONFIG["optimizer"]["type"] == "AdamW"

    def test_dpo_deepspeed_has_no_optimizer(self):
        from leap_finetune.training_configs.dpo_configs import DEEPSPEED_CONFIG

        assert "optimizer" not in DEEPSPEED_CONFIG

    def test_vlm_deepspeed_has_no_optimizer(self):
        from leap_finetune.training_configs.vlm_sft_config import DEEPSPEED_CONFIG

        assert "optimizer" not in DEEPSPEED_CONFIG

    def test_moe_sft_deepspeed_has_optimizer(self):
        from leap_finetune.training_configs.sft_configs import MOE_DEEPSPEED_CONFIG

        assert "optimizer" in MOE_DEEPSPEED_CONFIG
        assert MOE_DEEPSPEED_CONFIG["optimizer"]["type"] == "AdamW"

    def test_moe_dpo_deepspeed_has_no_optimizer(self):
        from leap_finetune.training_configs.dpo_configs import MOE_DEEPSPEED_CONFIG

        assert "optimizer" not in MOE_DEEPSPEED_CONFIG

    def test_sft_deepspeed_zero_stage_2(self):
        from leap_finetune.training_configs.sft_configs import DEEPSPEED_CONFIG

        assert DEEPSPEED_CONFIG["zero_optimization"]["stage"] == 2

    def test_moe_deepspeed_zero_stage_0(self):
        from leap_finetune.training_configs.sft_configs import MOE_DEEPSPEED_CONFIG

        assert MOE_DEEPSPEED_CONFIG["zero_optimization"]["stage"] == 0


# ============================================================================
# NEW: MoE detection tests
# ============================================================================


class TestMoEDetection:
    def test_dense_not_moe(self):
        from leap_finetune.utils.model_utils import is_moe_model_from_name

        assert not is_moe_model_from_name("LFM2-1.2B")

    def test_8b_a1b_is_moe(self):
        from leap_finetune.utils.model_utils import is_moe_model_from_name

        assert is_moe_model_from_name("LFM2-8B-A1B")

    def test_case_insensitive(self):
        from leap_finetune.utils.model_utils import is_moe_model_from_name

        assert is_moe_model_from_name("lfm2-8b-a1b")

    def test_moe_string_in_name(self):
        from leap_finetune.utils.model_utils import is_moe_model_from_name

        assert is_moe_model_from_name("some-moe-model")


# ============================================================================
# NEW: FSDP config tests
# ============================================================================


class TestFSDPConfig:
    def test_moe_fsdp_wraps_correct_layer(self):
        from leap_finetune.training_configs.distributed_configs import MOE_FSDP_CONFIG

        assert (
            MOE_FSDP_CONFIG["fsdp_config"]["transformer_layer_cls_to_wrap"]
            == "Lfm2MoeDecoderLayer"
        )

    def test_moe_fsdp_shard_grad_op(self):
        from leap_finetune.training_configs.distributed_configs import MOE_FSDP_CONFIG

        assert "shard_grad_op" in MOE_FSDP_CONFIG["fsdp"]
        assert "auto_wrap" in MOE_FSDP_CONFIG["fsdp"]
