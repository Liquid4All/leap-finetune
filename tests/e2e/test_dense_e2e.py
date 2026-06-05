from types import SimpleNamespace

import pyarrow as pa
import pytest
from torch.utils.data.distributed import DistributedSampler

from conftest import (
    assert_checkpoints_exist,
    assert_eval_callback_logged,
    assert_training_result,
    requires_gpu,
    run_e2e_training,
)
from leap_finetune.data_loading.ray_data_utils import ray_dataset_to_hf
from leap_finetune.training.sft import LFMSFTTrainer

pytestmark = pytest.mark.dense

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


# === Dense Ray sharding contract ===


class _RayShard:
    def __init__(self, rows):
        self._rows = rows

    def iter_batches(self, *, batch_format="pyarrow"):
        assert batch_format == "pyarrow"
        yield pa.Table.from_pylist(self._rows)


def _sample_id_collator(rows):
    return {"sample_id": [row["sample_id"] for row in rows]}


def _collect_sample_ids(dataloader):
    sample_ids = []
    for batch in dataloader:
        sample_ids.extend(batch["sample_id"])
    return sample_ids


class TestDenseRaySharding:
    def test_worker_shards_are_not_sharded_again_by_trainer_dataloader(self):
        rows = [{"sample_id": i} for i in range(8)]
        worker_shards = [
            ray_dataset_to_hf(_RayShard(rows[:4])),
            ray_dataset_to_hf(_RayShard(rows[4:])),
        ]

        consumed_by_worker = []
        for shard in worker_shards:
            trainer = LFMSFTTrainer.__new__(LFMSFTTrainer)
            trainer.train_dataset = shard
            trainer.args = SimpleNamespace(per_device_train_batch_size=2)
            trainer.data_collator = _sample_id_collator

            dataloader = trainer.get_train_dataloader()

            assert not isinstance(dataloader.sampler, DistributedSampler)
            consumed_by_worker.append(sorted(_collect_sample_ids(dataloader)))

        assert consumed_by_worker == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert sorted(sum(consumed_by_worker, [])) == list(range(8))


# === Dense SFT with LoRA ===


class TestDenseSFTLoRA:
    @requires_gpu
    def test_training_completes_and_learns(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_sft_lora.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result, max_eval_loss=7.0, check_loss_trend=False)
        assert_eval_callback_logged(result)


# === Dense SFT full fine-tune ===


class TestDenseSFTFull:
    @requires_gpu
    def test_training_completes_learns_and_checkpoints(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_sft_full.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result)
        assert_eval_callback_logged(result)

        assert_checkpoints_exist(e2e_output_dir)


# === Dense DPO with LoRA ===


class TestDenseDPOLoRA:
    @requires_gpu
    def test_training_completes_and_learns(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_dpo_lora.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result, check_loss_trend=False)
        assert_eval_callback_logged(result)


# === Dense DPO full fine-tune ===


class TestDenseDPOFull:
    @requires_gpu
    def test_training_completes_learns_and_checkpoints(self, e2e_output_dir):
        config_path = str(FIXTURES / "e2e_dpo_full.yaml")
        result = run_e2e_training(config_path, e2e_output_dir)
        assert_training_result(result, check_loss_trend=False)
        assert_eval_callback_logged(result)

        assert_checkpoints_exist(e2e_output_dir)
