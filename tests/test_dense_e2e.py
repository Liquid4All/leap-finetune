import pathlib

import pytest

from conftest import assert_training_result, requires_gpu, run_e2e_training

pytestmark = pytest.mark.dense

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# === Dense SFT with LoRA ===


class TestDenseSFTLoRA:
    @requires_gpu
    def test_training_completes_and_learns(self, tmp_path):
        config_path = str(FIXTURES / "e2e_sft_lora.yaml")
        result = run_e2e_training(config_path, tmp_path)
        assert_training_result(result)


# === Dense SFT full fine-tune ===


class TestDenseSFTFull:
    @requires_gpu
    def test_training_completes_and_learns(self, tmp_path):
        config_path = str(FIXTURES / "e2e_sft_full.yaml")
        result = run_e2e_training(config_path, tmp_path)
        assert_training_result(result)

    @requires_gpu
    def test_checkpoint_exists(self, tmp_path):
        config_path = str(FIXTURES / "e2e_sft_full.yaml")
        run_e2e_training(config_path, tmp_path)
        # save_strategy=epoch means checkpoint dirs should exist under output
        checkpoint_dirs = list(tmp_path.rglob("checkpoint-*"))
        assert len(checkpoint_dirs) > 0, "No checkpoint directories found"


# === Dense DPO with LoRA ===


class TestDenseDPOLoRA:
    @requires_gpu
    def test_training_completes_and_learns(self, tmp_path):
        config_path = str(FIXTURES / "e2e_dpo_lora.yaml")
        result = run_e2e_training(config_path, tmp_path)
        assert_training_result(result)


# === Dense DPO full fine-tune ===


class TestDenseDPOFull:
    @requires_gpu
    def test_training_completes_and_learns(self, tmp_path):
        config_path = str(FIXTURES / "e2e_dpo_full.yaml")
        result = run_e2e_training(config_path, tmp_path)
        assert_training_result(result)

    @requires_gpu
    def test_checkpoint_exists(self, tmp_path):
        config_path = str(FIXTURES / "e2e_dpo_full.yaml")
        run_e2e_training(config_path, tmp_path)
        checkpoint_dirs = list(tmp_path.rglob("checkpoint-*"))
        assert len(checkpoint_dirs) > 0, "No checkpoint directories found"
