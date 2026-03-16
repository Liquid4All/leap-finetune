import math
import os
import pathlib
import shutil

import pytest
import torch
import yaml

from leap_finetune.utils.constants import LEAP_FINETUNE_DIR

# === Ray temp dir ===
# On shared machines /tmp/ray may be owned by another user.
# Use /tmp/$USER/ray to avoid permission conflicts.
_RAY_TMPDIR = pathlib.Path(f"/tmp/{os.environ.get('USER', 'default')}/ray")
_RAY_TMPDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("RAY_TMPDIR", str(_RAY_TMPDIR))

# === E2E test output dir ===
# Use ~/test-results instead of /tmp to avoid filling the shared /tmp partition
# with large model checkpoints (MoE FSDP checkpoints are ~27GB each).
_TEST_RESULTS_DIR = pathlib.Path.home() / "test-results"


# === CLI flag registration ===


def pytest_addoption(parser):
    parser.addoption("--configs", action="store_true", help="Run only config tests")
    parser.addoption("--data", action="store_true", help="Run only data tests")
    parser.addoption("--dense", action="store_true", help="Run only dense GPU tests")
    parser.addoption("--vlm", action="store_true", help="Run only VLM GPU tests")
    parser.addoption("--moe", action="store_true", help="Run only MoE GPU tests")


def pytest_collection_modifyitems(config, items):
    flag_mark_map = {
        "configs": "configs",
        "data": "data",
        "dense": "dense",
        "vlm": "vlm",
        "moe": "moe",
    }
    active = [
        mark
        for flag, mark in flag_mark_map.items()
        if config.getoption(flag, default=False)
    ]

    if not active:
        return

    skip = pytest.mark.skip(reason="Not selected by CLI flags")
    for item in items:
        item_marks = {m.name for m in item.iter_markers()}
        if not item_marks.intersection(active):
            item.add_marker(skip)


# === Skip markers ===

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No GPU available",
)

requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires 2+ GPUs",
)


# === Shared fixtures ===


@pytest.fixture
def job_configs_dir():
    return LEAP_FINETUNE_DIR / "job_configs"


@pytest.fixture
def fixtures_dir():
    return pathlib.Path(__file__).parent / "fixtures"


BASE_SFT_DATASET = {
    "path": "HuggingFaceTB/smoltalk",
    "type": "sft",
    "limit": 10,
    "test_size": 0.2,
    "subset": "all",
}

BASE_DPO_DATASET = {
    "path": "mlabonne/orpo-dpo-mix-40k",
    "type": "dpo",
    "limit": 10,
    "test_size": 0.2,
    "subset": "default",
}


def write_config(config: dict, tmp_path: pathlib.Path) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


@pytest.fixture
def write_config_fn(tmp_path):
    def _write(config: dict) -> str:
        return write_config(config, tmp_path)

    return _write


# === E2E training helper ===


@pytest.fixture
def e2e_output_dir():
    """Provide ~/test-results as output dir, cleaned up after each test."""
    _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    yield _TEST_RESULTS_DIR
    shutil.rmtree(_TEST_RESULTS_DIR, ignore_errors=True)


def run_e2e_training(config_path: str, output_dir: pathlib.Path):
    """Parse config, override output_dir, run training, return Result."""
    os.environ["OUTPUT_DIR"] = str(output_dir)
    try:
        from leap_finetune.utils.config_parser import parse_job_config
        from leap_finetune.trainer import ray_trainer

        job_config = parse_job_config(config_path)
        job_config_dict = job_config.to_dict()
        return ray_trainer(job_config_dict)
    finally:
        os.environ.pop("OUTPUT_DIR", None)


def assert_training_result(result, max_eval_loss=5.0):
    """Verify training completed, produced finite loss, and actually learned.

    Args:
        result: Ray Train Result object.
        max_eval_loss: Upper bound on eval_loss. Random cross-entropy for
            vocab=65536 is ~11.1, so anything above max_eval_loss indicates
            the model didn't learn. Default 5.0 is generous for 1-epoch runs.
    """
    assert result is not None, "Training returned no result"
    metrics = result.metrics
    assert metrics is not None, "No metrics in training result"

    # Training must have run at least 1 epoch
    assert "epoch" in metrics, f"No epoch in metrics: {metrics}"

    # eval_loss must exist, be finite, and show the model learned
    assert "eval_loss" in metrics, f"No eval_loss in metrics: {metrics}"
    eval_loss = metrics["eval_loss"]
    assert math.isfinite(eval_loss), f"eval_loss is not finite: {eval_loss}"
    assert eval_loss < max_eval_loss, (
        f"eval_loss {eval_loss:.4f} >= {max_eval_loss} — model did not learn. "
        f"Random baseline for vocab=65536 is ~11.1"
    )

    # train_loss should also be present and finite
    if "train_loss" in metrics:
        assert math.isfinite(metrics["train_loss"]), (
            f"train_loss is not finite: {metrics['train_loss']}"
        )
