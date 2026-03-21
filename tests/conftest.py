import math
import os
import pathlib
import re
import shutil

import pytest
import torch
import yaml

from leap_finetune.utils.constants import LEAP_FINETUNE_DIR

# === Ray temp dir ===
_RAY_TMPDIR = pathlib.Path(f"/tmp/{os.environ.get('USER', 'default')}/ray")
_RAY_TMPDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("RAY_TMPDIR", str(_RAY_TMPDIR))

# === E2E test output dir ===

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
def sft_config_path(job_configs_dir):
    return str(job_configs_dir / "sft_example.yaml")


@pytest.fixture
def dpo_config_path(job_configs_dir):
    return str(job_configs_dir / "dpo_example.yaml")


@pytest.fixture
def vlm_config_path(job_configs_dir):
    return str(job_configs_dir / "vlm_sft_example.yaml")


@pytest.fixture
def moe_sft_config_path(job_configs_dir):
    return str(job_configs_dir / "moe_sft_example.yaml")


@pytest.fixture
def moe_dpo_config_path(job_configs_dir):
    return str(job_configs_dir / "moe_dpo_example.yaml")


@pytest.fixture
def slurm_config_path(job_configs_dir):
    return str(job_configs_dir / "sft_example_with_slurm.yaml")


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

BASE_VLM_DATASET = {
    "path": "alay2shah/example-vlm-sft-dataset",
    "type": "vlm_sft",
    "limit": 10,
    "test_size": 0.2,
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


def assert_training_result(
    result, max_eval_loss=5.0, check_loss_trend=True, check_dpo_preference=False
):
    """Verify training completed, produced finite loss, and optionally learning signals.

    Args:
        result: Ray Train Result object.
        max_eval_loss: Upper bound on eval_loss. Random cross-entropy for
            vocab=65536 is ~11.1, so anything above max_eval_loss indicates
            the model didn't learn. Default 5.0 is generous for 1-epoch runs.
        check_loss_trend: Whether to assert loss trends downward. Should be
            False for DPO — DPO loss measures preference margin, not
            cross-entropy, so it often stays flat or fluctuates even when
            the model is learning (eval_rewards/accuracies is the real signal).
        check_dpo_preference: Whether to assert DPO reward accuracy > 0.5 and
            positive reward margins. Use for DPO full fine-tune where the model
            has enough capacity and steps to learn preferences.
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

    # Check for loss trend downward from first and last quarter of training.
    if check_loss_trend:
        loss_history = metrics.get("loss_history", [])
        if len(loss_history) >= 4:
            q = max(1, len(loss_history) // 4)
            early_avg = sum(loss_history[:q]) / q
            late_avg = sum(loss_history[-q:]) / q
            assert late_avg < early_avg, (
                f"Loss did not trend down: "
                f"first quarter avg={early_avg:.4f} → last quarter avg={late_avg:.4f}"
            )

    # DPO-specific -- check for reward margin / acc over the loss trend.
    if check_dpo_preference:
        if "eval_rewards/accuracies" in metrics:
            acc = metrics["eval_rewards/accuracies"]
            assert acc > 0.5, (
                f"DPO eval reward accuracy {acc:.2f} <= 0.5 — "
                f"model is not preferring chosen over rejected"
            )
        if "eval_rewards/margins" in metrics:
            margin = metrics["eval_rewards/margins"]
            assert margin > 0.0, (
                f"DPO eval reward margin {margin:.4f} <= 0 — "
                f"chosen reward is not higher than rejected"
            )

    # train_loss should also be present and finite
    if "train_loss" in metrics:
        assert math.isfinite(metrics["train_loss"]), (
            f"train_loss is not finite: {metrics['train_loss']}"
        )


def assert_checkpoints_exist(output_dir: pathlib.Path):
    """Verify at least one checkpoint directory exists (original or renamed)."""
    checkpoint_dirs = list(output_dir.rglob("checkpoint-*"))
    renamed_dirs = [
        d
        for d in output_dir.iterdir()
        if d.is_dir() and re.search(r"-e\d+s\d+-", d.name)
    ]
    assert len(checkpoint_dirs) + len(renamed_dirs) > 0, (
        f"No checkpoint directories found under {output_dir}. "
        f"Contents: {[p.name for p in output_dir.iterdir()]}"
    )
