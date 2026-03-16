import pathlib

import pytest

from conftest import assert_training_result, requires_multi_gpu, run_e2e_training

pytestmark = pytest.mark.moe

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# === MoE SFT with LoRA (DeepSpeed stage 0 path) ===


class TestMoESFTLoRA:
    @requires_multi_gpu
    def test_training_completes_and_learns(self, tmp_path):
        config_path = str(FIXTURES / "e2e_moe_sft_lora.yaml")
        result = run_e2e_training(config_path, tmp_path)
        assert_training_result(result)


# === MoE SFT full fine-tune (FSDP path) ===


class TestMoESFTFull:
    @requires_multi_gpu
    def test_training_completes_and_learns(self, tmp_path):
        config_path = str(FIXTURES / "e2e_moe_sft_full.yaml")
        result = run_e2e_training(config_path, tmp_path)
        assert_training_result(result)

    @requires_multi_gpu
    def test_checkpoint_exists(self, tmp_path):
        config_path = str(FIXTURES / "e2e_moe_sft_full.yaml")
        run_e2e_training(config_path, tmp_path)
        checkpoint_dirs = list(tmp_path.rglob("checkpoint-*"))
        assert len(checkpoint_dirs) > 0, "No checkpoint directories found"
