import os
import pathlib
import tempfile

import pytest
import yaml

from leap_finetune.utils.checkpoint_callback import LeapCheckpointCallback
from leap_finetune.utils.config_parser import (
    generate_run_name,
    parse_job_config,
    resolve_config_path,
)
from leap_finetune.utils.constants import LEAP_FINETUNE_DIR


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
        # Dataset name should be truncated to 10 chars
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


class TestExtendsResolution:
    def test_extends_default_sft(self, sft_config_path):
        job = parse_job_config(sft_config_path)
        config = job.training_config.value
        # Should have the override values
        assert config["num_train_epochs"] == 3
        assert config["per_device_train_batch_size"] == 2
        assert config["learning_rate"] == 2e-5
        # Should preserve base values
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


class TestProjectNameFallback:
    def test_project_name_used(self, sft_config_path):
        job = parse_job_config(sft_config_path)
        assert job.job_name == "my_sft_project"

    def test_job_name_fallback(self):
        config = {
            "job_name": "test_job",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": {
                "path": "HuggingFaceTB/smoltalk",
                "type": "sft",
                "limit": 10,
                "test_size": 0.2,
                "subset": "all",
            },
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        try:
            yaml.dump(config, f)
            f.flush()
            job = parse_job_config(f.name)
            assert job.job_name == "test_job"
        finally:
            os.unlink(f.name)

    def test_default_fallback(self):
        config = {
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": {
                "path": "HuggingFaceTB/smoltalk",
                "type": "sft",
                "limit": 10,
                "test_size": 0.2,
                "subset": "all",
            },
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        try:
            yaml.dump(config, f)
            f.flush()
            job = parse_job_config(f.name)
            assert job.job_name == "default_job"
        finally:
            os.unlink(f.name)


class TestInvalidConfigs:
    def test_unknown_extends_raises(self):
        config = {
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": {
                "path": "HuggingFaceTB/smoltalk",
                "type": "sft",
                "limit": 10,
                "test_size": 0.2,
                "subset": "all",
            },
            "training_config": {"extends": "NONEXISTENT_CONFIG"},
        }
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        try:
            yaml.dump(config, f)
            f.flush()
            with pytest.raises(ValueError, match="Unknown base config"):
                parse_job_config(f.name)
        finally:
            os.unlink(f.name)

    def test_unknown_peft_extends_raises(self):
        config = {
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": {
                "path": "HuggingFaceTB/smoltalk",
                "type": "sft",
                "limit": 10,
                "test_size": 0.2,
                "subset": "all",
            },
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"extends": "NONEXISTENT_LORA", "use_peft": True},
        }
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        try:
            yaml.dump(config, f)
            f.flush()
            with pytest.raises(ValueError, match="Unknown base PEFT config"):
                parse_job_config(f.name)
        finally:
            os.unlink(f.name)

    def test_unknown_training_type_raises(self):
        config = {
            "model_name": "LFM2-1.2B",
            "training_type": "invalid_type",
            "dataset": {
                "path": "HuggingFaceTB/smoltalk",
                "type": "sft",
                "limit": 10,
                "test_size": 0.2,
                "subset": "all",
            },
        }
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        try:
            yaml.dump(config, f)
            f.flush()
            with pytest.raises(ValueError, match="Unknown training type"):
                parse_job_config(f.name)
        finally:
            os.unlink(f.name)


# === Callback instantiation ===


class TestLeapCheckpointCallback:
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
