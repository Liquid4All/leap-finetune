import pytest

from leap_finetune.config import materialize_job_config, parse_job_config
from leap_finetune.rl.rewards import resolve_reward_specs

from conftest import write_config

pytestmark = pytest.mark.configs


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


class TestGRPOSmoke:
    def test_minimal_grpo_materializes(self, tmp_path):
        config = {
            "project_name": "test_grpo",
            "model_name": "LFM2-1.2B",
            "training_type": "grpo",
            "dataset": GRPO_DATASET,
            "training_config": {},
            "rewards": {
                "funcs": ["./rewards/length.py::length_reward"],
                "weights": [1.0],
            },
        }
        parsed = parse_job_config(write_config(config, tmp_path))
        jc = materialize_job_config(parsed)
        assert jc.training_type == "grpo"
        assert jc.dataset.test_size == 0.01
        assert jc.rewards["weights"] == [1.0]
        assert jc.config_dir == str(tmp_path.resolve())

    def test_reward_paths_are_absolutized(self, tmp_path):
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
            "training_config": {},
            "rewards": {
                "funcs": ["./local_reward.py::local_reward"],
                "weights": [1.0],
            },
        }
        parsed = parse_job_config(write_config(config, tmp_path))
        jc = materialize_job_config(parsed)
        assert jc.rewards["funcs"] == [f"{reward_file.resolve()}::local_reward"]

    def test_judge_reward_config_persists(self, tmp_path):
        config = {
            "project_name": "t",
            "model_name": "LFM2-1.2B",
            "training_type": "grpo",
            "dataset": GRPO_DATASET,
            "training_config": {},
            "rewards": {
                "judge": {
                    "model": "LFM2-1.2B",
                    "base_url": "http://judge:8001",
                    "weight": 0.5,
                }
            },
        }
        parsed = parse_job_config(write_config(config, tmp_path))
        jc = materialize_job_config(parsed)
        funcs, weights = resolve_reward_specs(jc.rewards, tmp_path)
        assert [f.__name__ for f in funcs] == ["judge_llm_reward"]
        assert weights == [0.5]

    def test_vlm_grpo_materializes(self, tmp_path):
        config = {
            "project_name": "vlm_grpo_test",
            "model_name": "LFM2-VL-1.6B",
            "training_type": "vlm_grpo",
            "dataset": VLM_GRPO_DATASET,
            "training_config": {},
            "rl_env": {
                "source": "liquidai/vlm-grounding-bbox-env",
                "max_turns": 1,
            },
            "grpo_rollout": {"server_gpus": 1, "tensor_parallel_size": 1},
            "rewards": {
                "funcs": ["./rewards/length.py::length_reward"],
                "weights": [0.2],
            },
        }
        parsed = parse_job_config(write_config(config, tmp_path))
        jc = materialize_job_config(parsed)
        assert jc.training_type == "vlm_grpo"
        assert jc.rl_env["source"] == "liquidai/vlm-grounding-bbox-env"
        assert jc.grpo_rollout["server_gpus"] == 1
        assert jc.training_config["lr_multipliers"]["model.vision_tower"] == 0.1
