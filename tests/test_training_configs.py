import pytest

from leap_finetune.utils.config_parser import parse_job_config

from conftest import BASE_DPO_DATASET, BASE_SFT_DATASET, BASE_VLM_DATASET, write_config

pytestmark = pytest.mark.configs


# === Config pipeline integrity ===


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

        excluded = SFT_EXCLUDED_KEYS | {"leap_run_name_template", "deepspeed"}
        filtered = {k: v for k, v in d["training_config"].items() if k not in excluded}
        assert "deepspeed" not in filtered

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


# === DeepSpeed config structure ===


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


# === MoE detection ===


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


# === FSDP config ===


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
