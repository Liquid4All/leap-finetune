import pytest

pytestmark = pytest.mark.data


# === DatasetLoader construction ===


class TestDatasetLoader:
    def test_invalid_test_size_zero_raises(self):
        from leap_finetune.data_loaders.dataset_loader import DatasetLoader

        with pytest.raises(ValueError, match="test_size must be between"):
            DatasetLoader(dataset_path="x", dataset_type="sft", test_size=0)

    def test_invalid_test_size_above_one_raises(self):
        from leap_finetune.data_loaders.dataset_loader import DatasetLoader

        with pytest.raises(ValueError, match="test_size must be between"):
            DatasetLoader(dataset_path="x", dataset_type="sft", test_size=1.5)

    def test_valid_construction(self):
        from leap_finetune.data_loaders.dataset_loader import DatasetLoader

        loader = DatasetLoader(
            dataset_path="HuggingFaceTB/smoltalk",
            dataset_type="sft",
            subset="all",
            limit=20,
        )
        assert loader.dataset_type == "sft"
        assert loader.limit == 20
        assert loader.test_size == 0.2


# === SFT tokenization ===


class TestTokenizationSFT:
    @pytest.fixture(scope="class")
    def ray_session(self):
        import ray

        if not ray.is_initialized():
            ray.init(address="local", num_cpus=2)
        yield
        ray.shutdown()

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from leap_finetune.utils.load_models import load_tokenizer

        return load_tokenizer("LFM2-1.2B")

    @pytest.fixture(scope="class")
    def sft_datasets(self, ray_session, tokenizer):
        from leap_finetune.data_loaders.dataset_loader import DatasetLoader
        from leap_finetune.data_loaders.ray_data_utils import create_ray_datasets

        loader = DatasetLoader(
            dataset_path="HuggingFaceTB/smoltalk",
            dataset_type="sft",
            subset="all",
            limit=50,
        )
        train_ds, eval_ds = create_ray_datasets(
            loader,
            tokenizer=tokenizer,
            training_config={"max_length": 128, "packing": False},
        )
        return train_ds, eval_ds

    def test_sft_tokenization_columns(self, sft_datasets):
        train_ds, _ = sft_datasets
        row = next(iter(train_ds.iter_rows()))
        assert "input_ids" in row
        assert isinstance(row["input_ids"], list)
        assert all(isinstance(x, int) for x in row["input_ids"])

    def test_sft_tokenization_max_length_respected(self, sft_datasets):
        train_ds, _ = sft_datasets
        for row in train_ds.iter_rows():
            assert len(row["input_ids"]) <= 128

    def test_sft_packing_changes_row_count(self, ray_session, tokenizer):
        from leap_finetune.data_loaders.dataset_loader import DatasetLoader
        from leap_finetune.data_loaders.ray_data_utils import create_ray_datasets

        loader = DatasetLoader(
            dataset_path="HuggingFaceTB/smoltalk",
            dataset_type="sft",
            subset="all",
            limit=50,
        )
        train_unpacked, _ = create_ray_datasets(
            loader,
            tokenizer=tokenizer,
            training_config={"max_length": 256, "packing": False},
        )
        train_packed, _ = create_ray_datasets(
            loader,
            tokenizer=tokenizer,
            training_config={"max_length": 256, "packing": True},
        )
        unpacked_count = train_unpacked.count()
        packed_count = train_packed.count()
        assert unpacked_count != packed_count


# === DPO tokenization ===


class TestTokenizationDPO:
    @pytest.fixture(scope="class")
    def ray_session(self):
        import ray

        if not ray.is_initialized():
            ray.init(address="local", num_cpus=2)
        yield
        ray.shutdown()

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from leap_finetune.utils.load_models import load_tokenizer

        return load_tokenizer("LFM2-1.2B")

    @pytest.fixture(scope="class")
    def dpo_datasets(self, ray_session, tokenizer):
        from leap_finetune.data_loaders.dataset_loader import DatasetLoader
        from leap_finetune.data_loaders.ray_data_utils import create_ray_datasets

        loader = DatasetLoader(
            dataset_path="mlabonne/orpo-dpo-mix-40k",
            dataset_type="dpo",
            subset="default",
            limit=50,
        )
        train_ds, eval_ds = create_ray_datasets(
            loader,
            tokenizer=tokenizer,
            training_config={},
        )
        return train_ds, eval_ds

    def test_dpo_tokenization_columns(self, dpo_datasets):
        train_ds, _ = dpo_datasets
        row = next(iter(train_ds.iter_rows()))
        assert "prompt_input_ids" in row
        assert "chosen_input_ids" in row
        assert "rejected_input_ids" in row

    def test_dpo_eos_appended(self, dpo_datasets, tokenizer):
        train_ds, _ = dpo_datasets
        eos_id = tokenizer.eos_token_id
        row = next(iter(train_ds.iter_rows()))
        assert row["chosen_input_ids"][-1] == eos_id
        assert row["rejected_input_ids"][-1] == eos_id


# === Sharding correctness ===


class TestShardingCorrectness:
    @pytest.fixture(scope="class")
    def ray_session(self):
        import ray

        if not ray.is_initialized():
            ray.init(address="local", num_cpus=2)
        yield
        ray.shutdown()

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from leap_finetune.utils.load_models import load_tokenizer

        return load_tokenizer("LFM2-1.2B")

    @pytest.fixture(scope="class")
    def tokenized_train_ds(self, ray_session, tokenizer):
        from leap_finetune.data_loaders.dataset_loader import DatasetLoader
        from leap_finetune.data_loaders.ray_data_utils import create_ray_datasets

        loader = DatasetLoader(
            dataset_path="HuggingFaceTB/smoltalk",
            dataset_type="sft",
            subset="all",
            limit=100,
        )
        train_ds, _ = create_ray_datasets(
            loader,
            tokenizer=tokenizer,
            training_config={"max_length": 128, "packing": False},
        )
        return train_ds

    def test_no_duplicate_rows_across_shards(self, tokenized_train_ds):
        shards = tokenized_train_ds.split(2)
        shard_0_ids = [tuple(row["input_ids"]) for row in shards[0].iter_rows()]
        shard_1_ids = [tuple(row["input_ids"]) for row in shards[1].iter_rows()]
        overlap = set(shard_0_ids) & set(shard_1_ids)
        assert len(overlap) == 0, f"Found {len(overlap)} duplicate rows across shards"

    def test_shard_sizes_roughly_equal(self, tokenized_train_ds):
        shards = tokenized_train_ds.split(2)
        sizes = [shard.count() for shard in shards]
        assert abs(sizes[0] - sizes[1]) <= 1


# === Custom trainer DataLoader bypass ===


class TestCustomTrainerDataLoaders:
    def test_lfm_sft_trainer_no_distributed_sampler(self):
        from unittest.mock import MagicMock

        from datasets import Dataset
        from torch.utils.data import DistributedSampler
        from transformers import TrainingArguments

        from leap_finetune.training_loops.sft_run import LFMSFTTrainer

        dummy_dataset = Dataset.from_dict(
            {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        )
        args = TrainingArguments(
            output_dir="/tmp/test_sft",
            per_device_train_batch_size=1,
            report_to="none",
            no_cuda=True,
        )
        model = MagicMock()
        model.config = MagicMock()
        trainer = LFMSFTTrainer(
            model=model,
            args=args,
            train_dataset=dummy_dataset,
            data_collator=lambda x: x,
        )
        dl = trainer.get_train_dataloader()
        assert not isinstance(dl.sampler, DistributedSampler)

    def test_lfm_dpo_trainer_skips_prepare_dataset(self):
        from datasets import Dataset

        from leap_finetune.training_loops.dpo_run import LFMDPOTrainer

        dummy = Dataset.from_dict({"a": [1, 2, 3]})
        result = LFMDPOTrainer._prepare_dataset(None, dummy)
        assert result is dummy
