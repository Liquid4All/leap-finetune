"""End-to-end KTO training smoke test (GPU required)."""

import pytest

from conftest import assert_training_result, requires_gpu, run_e2e_training

pytestmark = pytest.mark.dense

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


@requires_gpu
def test_kto_training_completes_and_evaluates(e2e_output_dir):
    result = run_e2e_training(str(FIXTURES / "e2e_kto.yaml"), e2e_output_dir)
    assert_training_result(result, max_eval_loss=5.0, check_loss_trend=False)
