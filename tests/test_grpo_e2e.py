"""End-to-end GRPO training smoke tests (GPU required).

These tests verify the whole pipeline: dataset loading, reward resolution,
vLLM colocate instantiation, one optimizer step, and metric logging.
They deliberately use tiny settings (num_generations=2, 16 samples,
max_completion_length=16) so they finish in under ~5 minutes on 1 H100.

Run on a GPU node with:
    uv run pytest --dense tests/test_grpo_e2e.py -v

Or via the supplied SLURM script:
    sbatch tests/grpo_e2e_sbatch.sh
"""

import math
import pathlib

import pytest

from conftest import (
    assert_checkpoints_exist,
    requires_gpu,
    run_e2e_training,
)

pytestmark = pytest.mark.dense

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _assert_grpo_result(result):
    """GRPO has no eval_loss (eval_strategy=no), so we assert on train_loss
    and at least one reward/ metric showing the loop actually ran."""
    assert result is not None, "Training returned no result"
    metrics = result.metrics
    assert metrics is not None, "No metrics in training result"

    # Training made at least one step
    assert metrics.get("epoch", 0) > 0, f"No epoch progress: {metrics}"

    # train_loss is finite
    if "train_loss" in metrics:
        assert math.isfinite(metrics["train_loss"]), (
            f"train_loss is not finite: {metrics['train_loss']}"
        )

    # At least one GRPO reward metric should be logged. TRL logs 'reward/<name>/mean'
    reward_keys = [k for k in metrics if k.startswith("reward")]
    assert reward_keys, f"No reward/ metrics logged. All metrics: {list(metrics)}"


class TestDenseGRPO:
    @requires_gpu
    def test_text_grpo_completes_one_step(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_grpo.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        _assert_grpo_result(result)
