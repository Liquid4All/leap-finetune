import pytest

from conftest import (
    assert_checkpoints_exist,
    assert_training_result,
    requires_gpu,
    run_e2e_training,
)

pytestmark = pytest.mark.dense

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


# === Dense SFT with LoRA ===


class TestDenseSFTLoRA:
    @requires_gpu
    def test_training_completes_and_learns(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_sft_lora.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result)


# === Dense SFT full fine-tune ===


class TestDenseSFTFull:
    @requires_gpu
    def test_training_completes_learns_and_checkpoints(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_sft_full.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result)

        assert_checkpoints_exist(e2e_output_dir)


# === Dense DPO with LoRA ===


class TestDenseDPOLoRA:
    @requires_gpu
    def test_training_completes_and_learns(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_dpo_lora.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result, check_loss_trend=False)


# === Dense DPO full fine-tune ===


class TestDenseDPOFull:
    @requires_gpu
    def test_training_completes_learns_and_checkpoints(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_dpo_full.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(
            result, check_loss_trend=False, check_dpo_preference=True
        )

        assert_checkpoints_exist(e2e_output_dir)
