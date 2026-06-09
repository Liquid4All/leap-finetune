import pytest

from leap_finetune.rl.rewards import resolve_reward_specs
from leap_finetune.config import (
    materialize_job_config,
    parse_job_config as _parse_job_config,
)
from leap_finetune.training.default_configs import TRAINING_DEFAULTS

from conftest import write_config

pytestmark = pytest.mark.configs


def parse_job_config(config_input):
    return materialize_job_config(_parse_job_config(config_input))


# === Shared fixtures ===


GRPO_DATASET = {
    "path": "trl-lib/DeepMath-103K",
    "type": "grpo",
    "limit": 10,
}

VLM_GRPO_DATASET = {
    "path": "alay2shah/example-vlm-sft-dataset",
    "type": "vlm_grpo",
    "limit": 10,
    "image_root": "/tmp/images",
}


# === Base config auto-discovery ===


class TestGRPOBaseConfigDiscovery:
    def test_default_grpo_discovered(self):
        assert "DEFAULT_GRPO" in TRAINING_DEFAULTS

    def test_default_vlm_grpo_discovered(self):
        assert "DEFAULT_VLM_GRPO" in TRAINING_DEFAULTS

    def test_moe_grpo_discovered(self):
        assert "MOE_GRPO" in TRAINING_DEFAULTS

    def test_default_grpo_has_trl_v1_fields(self):
        cfg = TRAINING_DEFAULTS["DEFAULT_GRPO"]
        assert cfg["training_type"] == "grpo"
        assert cfg["loss_type"] == "dapo"  # TRL v1 default
        assert cfg["beta"] == 0.0  # KL off by default
        assert cfg["vllm_mode"] == "colocate"
        assert cfg["use_vllm"] is True
        assert cfg["num_generations"] == 8

    def test_default_vlm_grpo_has_correct_training_type(self):
        cfg = TRAINING_DEFAULTS["DEFAULT_VLM_GRPO"]
        assert cfg["training_type"] == "vlm_grpo"
        assert cfg["num_generations"] == 4  # smaller groups for VLM memory
        assert "lr_multipliers" in cfg
        assert cfg["lr_multipliers"]["model.vision_tower"] == 0.1


# === Basic parse ===


class TestParseGRPOConfig:
    def test_minimal_grpo_config(self, tmp_path):
        config = {
            "project_name": "test_grpo",
            "model_name": "LFM2-1.2B",
            "training_type": "grpo",
            "dataset": GRPO_DATASET,
            "training_config": {"extends": "DEFAULT_GRPO"},
            "rewards": {
                "funcs": ["./rewards/length.py::length_reward"],
                "weights": [1.0],
            },
        }
        jc = parse_job_config(write_config(config, tmp_path))
        assert jc.training_type == "grpo"
        # Paths are pre-resolved to absolute by config_parser so Ray workers
        # (whose CWD is a sandbox) can find the reward files.
        assert len(jc.rewards["funcs"]) == 1
        assert jc.rewards["funcs"][0].endswith("rewards/length.py::length_reward")
        assert jc.rewards["weights"] == [1.0]
        assert jc.rl_env is None
        assert jc.grpo_rollout is None
        assert jc.config_dir == str(tmp_path.resolve())

    def test_config_relative_custom_reward_path_is_absolutized(self, tmp_path):
        reward_file = tmp_path / "local_reward.py"
        reward_file.write_text(
            "def local_reward(completions, **kwargs):\n"
            "    return [1.0 for _ in completions]\n"
        )
        config = {
            "project_name": "test_grpo",
            "model_name": "LFM2-1.2B",
            "training_type": "grpo",
            "dataset": GRPO_DATASET,
            "training_config": {"extends": "DEFAULT_GRPO"},
            "rewards": {
                "funcs": ["./local_reward.py::local_reward"],
                "weights": [1.0],
            },
        }
        jc = parse_job_config(write_config(config, tmp_path))

        assert jc.rewards["funcs"] == [f"{reward_file.resolve()}::local_reward"]

    def test_grpo_defaults_test_size_to_tiny(self, tmp_path):
        """GRPO is online so it doesn't need offline eval. We default test_size
        to 0.01 to keep the ray pipeline happy without wasting data."""
        config = {
            "project_name": "t",
            "model_name": "LFM2-1.2B",
            "training_type": "grpo",
            "dataset": {"path": "trl-lib/DeepMath-103K", "type": "grpo"},
            "rewards": ["./rewards/length.py::length_reward"],
        }
        jc = parse_job_config(write_config(config, tmp_path))
        assert jc.dataset.test_size == 0.01

    def test_grpo_override_persists(self, tmp_path):
        config = {
            "project_name": "t",
            "model_name": "LFM2-1.2B",
            "training_type": "grpo",
            "dataset": GRPO_DATASET,
            "training_config": {
                "extends": "DEFAULT_GRPO",
                "num_generations": 4,
                "learning_rate": 5e-7,
            },
            "rewards": ["./rewards/length.py::length_reward"],
        }
        jc = parse_job_config(write_config(config, tmp_path))
        cfg = jc.training_config.value
        assert cfg["num_generations"] == 4
        assert cfg["learning_rate"] == 5e-7
        # Inherited defaults still there
        assert cfg["loss_type"] == "dapo"
        assert cfg["vllm_mode"] == "colocate"

    def test_grpo_judge_reward_config_persists(self, tmp_path):
        config = {
            "project_name": "t",
            "model_name": "LFM2-1.2B",
            "training_type": "grpo",
            "dataset": GRPO_DATASET,
            "training_config": {"extends": "DEFAULT_GRPO"},
            "rewards": {
                "judge": {
                    "model": "LFM2-1.2B",
                    "base_url": "http://judge:8001",
                    "weight": 0.5,
                }
            },
        }
        jc = parse_job_config(write_config(config, tmp_path))
        assert jc.rewards["judge"]["model"] == "LFM2-1.2B"
        funcs, weights = resolve_reward_specs(jc.rewards, tmp_path)
        assert [f.__name__ for f in funcs] == ["judge_llm_reward"]
        assert weights == [0.5]


# === VLM GRPO ===


class TestParseVLMGRPOConfig:
    def test_full_vlm_grpo_with_rl_env_and_rollout(self, tmp_path):
        config = {
            "project_name": "vlm_grpo_test",
            "model_name": "LFM2-VL-1.6B",
            "training_type": "vlm_grpo",
            "dataset": VLM_GRPO_DATASET,
            "training_config": {"extends": "DEFAULT_VLM_GRPO"},
            "rl_env": {
                "source": "liquidai/vlm-grounding-bbox-env",
                "max_turns": 1,
            },
            "grpo_rollout": {
                "server_gpus": 1,
                "tensor_parallel_size": 1,
                "dtype": "bfloat16",
            },
            "rewards": {
                "funcs": ["./rewards/length.py::length_reward"],
                "weights": [0.2],
            },
        }
        jc = parse_job_config(write_config(config, tmp_path))
        assert jc.training_type == "vlm_grpo"
        assert jc.rl_env["source"] == "liquidai/vlm-grounding-bbox-env"
        assert jc.grpo_rollout["server_gpus"] == 1
        # Per-component LR multipliers come through
        cfg = jc.training_config.value
        assert "lr_multipliers" in cfg
        assert cfg["lr_multipliers"]["model.vision_tower"] == 0.1
        assert cfg["training_type"] == "vlm_grpo"

    def test_vision_encoder_lr_multiplier_override(self, tmp_path):
        config = {
            "project_name": "t",
            "model_name": "LFM2-VL-1.6B",
            "training_type": "vlm_grpo",
            "dataset": VLM_GRPO_DATASET,
            "training_config": {
                "extends": "DEFAULT_VLM_GRPO",
                "vision_encoder_lr_multiplier": 0.05,
            },
            "rewards": ["./rewards/length.py::length_reward"],
        }
        jc = parse_job_config(write_config(config, tmp_path))
        assert jc.training_config.value["vision_encoder_lr_multiplier"] == 0.05


# === Rejection of GRPO keys on non-GRPO runs ===


class TestGRPOKeysRejectedOnNonGRPO:
    def test_rewards_on_sft_rejected(self, tmp_path):
        config = {
            "project_name": "t",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": {"path": "x", "type": "sft"},
            "rewards": ["./rewards/length.py::length_reward"],
        }
        with pytest.raises((ValueError, Exception), match="grpo"):
            parse_job_config(write_config(config, tmp_path))

    def test_rl_env_on_dpo_rejected(self, tmp_path):
        config = {
            "project_name": "t",
            "model_name": "LFM2-1.2B",
            "training_type": "dpo",
            "dataset": {"path": "x", "type": "dpo"},
            "rl_env": {"source": "qgallouedec/echo_env"},
        }
        with pytest.raises((ValueError, Exception), match="grpo"):
            parse_job_config(write_config(config, tmp_path))

    def test_grpo_rollout_on_vlm_sft_rejected(self, tmp_path):
        config = {
            "project_name": "t",
            "model_name": "LFM2-VL-1.6B",
            "training_type": "vlm_sft",
            "dataset": {"path": "x", "type": "vlm_sft"},
            "grpo_rollout": {"server_gpus": 1},
        }
        with pytest.raises((ValueError, Exception), match="grpo"):
            parse_job_config(write_config(config, tmp_path))


# === Shipped example YAMLs ===


class TestShippedExampleConfigs:
    """The example YAMLs in job_configs/ must all parse cleanly."""

    @pytest.mark.parametrize(
        "filename",
        [
            "grpo_example.yaml",
            "grpo_server_mode_example.yaml",
            "vlm_grpo_grounding_example.yaml",
        ],
    )
    def test_example_parses(self, filename, job_configs_dir):
        path = str(job_configs_dir / filename)
        jc = parse_job_config(path)
        assert jc.training_type in ("grpo", "vlm_grpo")
        resolve_reward_specs(jc.rewards, job_configs_dir)

    @pytest.mark.parametrize(
        "filename",
        [
            "e2e_grpo.yaml",
            "e2e_vlm_grpo.yaml",
        ],
    )
    def test_grpo_smoke_fixtures_keep_vllm_enabled(self, filename, fixtures_dir):
        path = str(fixtures_dir / filename)
        jc = parse_job_config(path)
        assert jc.training_type in ("grpo", "vlm_grpo")
        assert jc.training_config.value["use_vllm"] is True
        assert jc.training_config.value["vllm_mode"] == "colocate"
