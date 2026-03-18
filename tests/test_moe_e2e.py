import pytest

from conftest import (
    assert_checkpoints_exist,
    assert_training_result,
    requires_multi_gpu,
    run_e2e_training,
)

pytestmark = pytest.mark.moe

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


# === MoE SFT with LoRA (DeepSpeed stage 0 path) ===


class TestMoESFTLoRA:
    @requires_multi_gpu
    def test_training_completes_and_learns(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_moe_sft_lora.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result)


# === MoE SFT full fine-tune (FSDP path) ===


class TestMoESFTFull:
    @requires_multi_gpu
    def test_training_completes_learns_and_checkpoints(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_moe_sft_full.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result)

        assert_checkpoints_exist(e2e_output_dir)
