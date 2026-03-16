import math
import os
import pathlib

import pytest
import torch
import yaml

from leap_finetune.utils.constants import LEAP_FINETUNE_DIR


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


def assert_training_result(result):
    """Common assertions for e2e training results."""
    assert result is not None, "Training returned no result"
    metrics = result.metrics
    assert metrics is not None, "No metrics in training result"

    # Loss should be finite
    if "eval_loss" in metrics:
        eval_loss = metrics["eval_loss"]
        assert math.isfinite(eval_loss), f"eval_loss is not finite: {eval_loss}"
