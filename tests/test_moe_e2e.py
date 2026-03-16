import re

import pytest

from conftest import assert_training_result, requires_multi_gpu, run_e2e_training

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

        checkpoint_dirs = list(e2e_output_dir.rglob("checkpoint-*"))
        renamed_dirs = [
            d
            for d in e2e_output_dir.iterdir()
            if d.is_dir() and re.search(r"-e\d+s\d+-", d.name)
        ]
        assert len(checkpoint_dirs) + len(renamed_dirs) > 0, (
            f"No checkpoint directories found under {e2e_output_dir}. "
            f"Contents: {[p.name for p in e2e_output_dir.iterdir()]}"
        )
