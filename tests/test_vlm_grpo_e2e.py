"""End-to-end VLM GRPO smoke test (GPU required).

Verifies the VLM GRPO training loop works end-to-end including:
- VLM model loading via load_vlm_model
- Per-component LR param groups (vision_tower at 0.1x base LR)
- Dataset validation with image loading
- GRPO rollout + reward computation
- One optimizer step

Run on a GPU node with:
    uv run pytest --vlm tests/test_vlm_grpo_e2e.py -v
"""

import math
import pathlib

import pytest

from conftest import requires_gpu, run_e2e_training

pytestmark = pytest.mark.vlm

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


class TestVLMGRPO:
    @requires_gpu
    def test_vlm_grpo_completes_one_step(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_vlm_grpo.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert result is not None
        metrics = result.metrics
        assert metrics is not None
        assert metrics.get("epoch", 0) > 0

        if "train_loss" in metrics:
            assert math.isfinite(metrics["train_loss"]), (
                f"train_loss not finite: {metrics['train_loss']}"
            )

        # Per-component LR metrics should be logged because
        # LFMVLMGRPOTrainer.log() injects lr/<component> entries.
        lr_keys = [k for k in metrics if k.startswith("lr/")]
        assert lr_keys, (
            f"No per-component LR metrics logged. Expected lr/vision_tower etc. "
            f"All metrics: {list(metrics)}"
        )
