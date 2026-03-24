import pathlib

import pytest
import yaml
from peft import LoraConfig

from leap_finetune.utils.config_parser import (
    generate_run_name,
    parse_job_config,
    resolve_config_path,
)
from leap_finetune.utils.constants import LEAP_FINETUNE_DIR

from conftest import BASE_DPO_DATASET, BASE_SFT_DATASET, BASE_VLM_DATASET, write_config

pytestmark = pytest.mark.configs


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
        assert job.dataset.dataset_path == "HuggingFaceTB/smoltalk"
        assert job.dataset.dataset_type == "sft"
        assert "deepspeed" in job.training_config.value
        assert "learning_rate" in job.training_config.value

    def test_parse_dpo_example(self, dpo_config_path):
        job = parse_job_config(dpo_config_path)
        assert job.training_type == "dpo"
        assert job.job_name == "my_dpo_project"
        assert job.dataset.dataset_type == "dpo"
        assert "beta" in job.training_config.value
        assert "deepspeed" in job.training_config.value

    def test_parse_vlm_example(self, vlm_config_path):
        job = parse_job_config(vlm_config_path)
        assert job.training_type == "vlm_sft"
        assert job.job_name == "my_vlm_project"
        assert job.dataset.dataset_type == "vlm_sft"
        assert "deepspeed" in job.training_config.value

    def test_parse_moe_sft_example(self, moe_sft_config_path):
        job = parse_job_config(moe_sft_config_path)
        assert job.training_type == "sft"
        assert job.model_name == "LFM2-8B-A1B"
        assert job.dataset.dataset_type == "sft"
        ds_config = job.training_config.value.get("deepspeed", {})
        assert ds_config.get("zero_optimization", {}).get("stage") == 0

    def test_parse_moe_dpo_example(self, moe_dpo_config_path):
        job = parse_job_config(moe_dpo_config_path)
        assert job.training_type == "dpo"
        assert job.model_name == "LFM2-8B-A1B"
        assert job.dataset.dataset_type == "dpo"


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

    EXPECTED = {
        "sft_example.yaml": {"type": "sft", "model": "LFM2-1.2B", "has_peft": True},
        "dpo_example.yaml": {"type": "dpo", "model": "LFM2-1.2B", "has_peft": True},
        "vlm_sft_example.yaml": {
            "type": "vlm_sft",
            "model": "LFM2-1.2B",
            "has_peft": True,
        },
        "moe_sft_example.yaml": {
            "type": "sft",
            "model": "LFM2-8B-A1B",
            "has_peft": True,
        },
        "moe_dpo_example.yaml": {
            "type": "dpo",
            "model": "LFM2-8B-A1B",
            "has_peft": True,
        },
        "sft_example_with_slurm.yaml": {
            "type": "sft",
            "model": "LFM2-1.2B",
            "has_peft": True,
        },
        "sft_with_lora_example.yaml": {
            "type": "sft",
            "model": "LFM2-1.2B",
            "has_peft": True,
        },
    }

    @pytest.mark.parametrize("config_name", list(EXPECTED.keys()))
    def test_parse_and_to_dict(self, job_configs_dir, config_name):
        config_path = str(job_configs_dir / config_name)
        job = parse_job_config(config_path)
        expected = self.EXPECTED[config_name]

        d = job.to_dict()
        assert isinstance(d["training_config"], dict)
        assert d["training_type"] == expected["type"]
        assert d["model_name"] == expected["model"]

        from leap_finetune.data_loaders.dataset_loader import DatasetLoader

        assert isinstance(d["dataset"], DatasetLoader)
        assert d["dataset"].dataset_path, "dataset_path is empty"

        tc = d["training_config"]
        assert "output_dir" in tc
        assert "learning_rate" in tc
        assert isinstance(tc["learning_rate"], float)

        if expected["has_peft"]:
            assert d["peft_config"] is not None, (
                f"{config_name} should have peft_config"
            )


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
        output_dir = job.training_config.value["output_dir"]
        assert output_dir.startswith(env_dir), (
            f"output_dir should be under OUTPUT_DIR: {output_dir}"
        )

    def test_dataset_path_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATASET_PATH", "custom/override/dataset")
        config = {
            "project_name": "test_ds_override",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        assert job.dataset.dataset_path == "custom/override/dataset"

    def test_dataset_path_falls_back_to_yaml(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DATASET_PATH", raising=False)
        config = {
            "project_name": "test_ds_fallback",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        assert job.dataset.dataset_path == BASE_SFT_DATASET["path"]


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


# === Benchmark config passthrough ===


class TestBenchmarkConfig:
    def test_benchmarks_parsed_from_yaml(self, tmp_path):
        config = {
            "project_name": "test_bench",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
            "benchmarks": {
                "max_new_tokens": 64,
                "benchmarks": [
                    {
                        "name": "eval1",
                        "path": "/data/eval.jsonl",
                        "metric": "short_answer",
                    },
                ],
            },
        }
        job = parse_job_config(write_config(config, tmp_path))
        assert job.benchmark_configs is not None
        assert len(job.benchmark_configs["benchmarks"]) == 1
        assert job.benchmark_configs["max_new_tokens"] == 64

    def test_no_benchmarks_gives_none(self, tmp_path):
        config = {
            "project_name": "test_no_bench",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
        }
        job = parse_job_config(write_config(config, tmp_path))
        assert job.benchmark_configs is None

    def test_benchmarks_in_to_dict(self, tmp_path):
        config = {
            "project_name": "test_bench_dict",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"extends": "DEFAULT_SFT"},
            "peft_config": {"use_peft": False},
            "benchmarks": {
                "benchmarks": [
                    {"name": "e", "path": "/data/e.jsonl", "metric": "mcq_gen"},
                ],
            },
        }
        job = parse_job_config(write_config(config, tmp_path))
        d = job.to_dict()
        assert "benchmark_configs" in d
        assert d["benchmark_configs"]["benchmarks"][0]["name"] == "e"


# === SLURM generation ===


class TestSlurmGeneration:
    def test_slurm_config_parsed(self, slurm_config_path):
        with open(slurm_config_path) as f:
            config = yaml.safe_load(f)
        assert "slurm" in config
        assert config["slurm"]["job_name"] == "sft_training"
        assert config["slurm"]["gpus_per_task"] == 4

    def test_generate_slurm_script(self, slurm_config_path):
        import tempfile

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
