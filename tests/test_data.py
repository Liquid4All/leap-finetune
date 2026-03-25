import pytest

from datasets import Dataset

pytestmark = pytest.mark.data


# === Tool call validation ===


class TestToolCallValidation:
    # --- SFT ---

    def test_sft_foreign_markers_rejected(self):
        from leap_finetune.data_loaders.validate_loader import validate_sft_format

        for marker, fmt in [("<tool_call>", "Qwen"), ("[TOOL_CALLS]", "Mistral")]:
            ds = Dataset.from_list(
                [
                    {
                        "messages": [
                            {"role": "user", "content": "Hi"},
                            {
                                "role": "assistant",
                                "content": f"{marker} foo {marker.replace('<', '</')}",
                            },
                        ]
                    }
                ]
            )
            with pytest.raises(ValueError, match=f"{fmt} tool call markers"):
                validate_sft_format(ds)

    def test_sft_correct_lfm_format_passes(self):
        from leap_finetune.data_loaders.validate_loader import validate_sft_format

        ds = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Get weather"},
                        {
                            "role": "assistant",
                            "content": '<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>\nChecking now.',
                        },
                        {"role": "tool", "content": '{"temp": 18}'},
                        {"role": "assistant", "content": "It's 18 degrees."},
                    ]
                }
            ]
        )
        assert len(validate_sft_format(ds)) == 1

    def test_sft_text_before_tool_call_rejected(self):
        from leap_finetune.data_loaders.validate_loader import validate_sft_format

        ds = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Get weather"},
                        {
                            "role": "assistant",
                            "content": 'Sure thing!\n<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>',
                        },
                        {"role": "tool", "content": '{"temp": 18}'},
                    ]
                }
            ]
        )
        with pytest.raises(ValueError, match="Text appears before tool call"):
            validate_sft_format(ds)

    def test_sft_structured_tool_calls_field_rejected(self):
        from leap_finetune.data_loaders.validate_loader import validate_sft_format

        ds = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {
                            "role": "assistant",
                            "content": "Let me check.",
                            "tool_calls": [{"name": "get_weather"}],
                        },
                    ]
                }
            ]
        )
        with pytest.raises(ValueError, match="tool_calls.*not supported"):
            validate_sft_format(ds)

    def test_sft_missing_tool_response_rejected(self):
        from leap_finetune.data_loaders.validate_loader import validate_sft_format

        ds = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Get weather"},
                        {
                            "role": "assistant",
                            "content": '<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>',
                        },
                        {"role": "user", "content": "Thanks"},
                    ]
                }
            ]
        )
        with pytest.raises(ValueError, match="no tool response"):
            validate_sft_format(ds)

    # --- DPO ---

    def test_dpo_foreign_markers_rejected(self):
        from leap_finetune.data_loaders.validate_loader import validate_dpo_format

        # String format
        ds = Dataset.from_list(
            [{"chosen": "<tool_call>foo</tool_call>", "rejected": "no tools"}]
        )
        with pytest.raises(ValueError, match="Qwen"):
            validate_dpo_format(ds)

        # Message list format
        ds = Dataset.from_list(
            [
                {
                    "chosen": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "<tool_call>foo</tool_call>"},
                    ],
                    "rejected": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "No tools."},
                    ],
                }
            ]
        )
        with pytest.raises(ValueError, match="Qwen"):
            validate_dpo_format(ds)

    def test_dpo_text_before_tool_call_rejected(self):
        from leap_finetune.data_loaders.validate_loader import validate_dpo_format

        ds = Dataset.from_list(
            [
                {
                    "chosen": 'Sure!\n<|tool_call_start|>[get_weather(location="SF")]<|tool_call_end|>',
                    "rejected": "I can't do that.",
                }
            ]
        )
        with pytest.raises(ValueError, match="Text appears before tool call"):
            validate_dpo_format(ds)

    # --- Row filters ---

    def test_sft_row_filter_rejects_bad_tool_calls(self):
        from leap_finetune.data_loaders.validate_loader import get_row_filter

        f = get_row_filter("sft")
        assert (
            f(
                {
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "<tool_call>x</tool_call>"},
                    ]
                }
            )
            is False
        )
        assert (
            f(
                {
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "ok", "tool_calls": [{}]},
                    ]
                }
            )
            is False
        )

    def test_dpo_row_filter_rejects_foreign_markers(self):
        from leap_finetune.data_loaders.validate_loader import get_row_filter

        f = get_row_filter("dpo")
        assert f({"chosen": "<tool_call>x</tool_call>", "rejected": "no"}) is False

    # --- Regression ---

    def test_non_tool_call_data_unaffected(self):
        from leap_finetune.data_loaders.validate_loader import (
            validate_dpo_format,
            validate_sft_format,
        )

        sft = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello!"},
                    ]
                }
            ]
        )
        assert len(validate_sft_format(sft)) == 1

        dpo = Dataset.from_list(
            [
                {
                    "chosen": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello!"},
                    ],
                    "rejected": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hey."},
                    ],
                }
            ]
        )
        assert len(validate_dpo_format(dpo)) == 1


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
            ray.init(address="local", num_cpus=2, runtime_env={"working_dir": None})
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

    def test_sft_tokenization_content(self, sft_datasets, tokenizer):
        train_ds, _ = sft_datasets
        row = next(iter(train_ds.iter_rows()))
        assert "input_ids" in row
        assert isinstance(row["input_ids"], list)
        assert len(row["input_ids"]) > 0, "input_ids is empty"
        assert all(isinstance(x, int) for x in row["input_ids"])

        # All token IDs must be valid (within vocab range)
        vocab_size = tokenizer.vocab_size
        for token_id in row["input_ids"]:
            assert 0 <= token_id < vocab_size, (
                f"Token ID {token_id} out of vocab range [0, {vocab_size})"
            )

    def test_sft_tokenization_max_length_respected(self, sft_datasets):
        train_ds, _ = sft_datasets
        for row in train_ds.iter_rows():
            assert len(row["input_ids"]) <= 128

    def test_sft_train_eval_split(self, sft_datasets):
        train_ds, eval_ds = sft_datasets
        train_count = train_ds.count()
        eval_count = eval_ds.count()
        assert train_count > 0, "Train dataset is empty"
        assert eval_count > 0, "Eval dataset is empty"
        # test_size=0.2 → eval should be ~20% of total
        total = train_count + eval_count
        eval_ratio = eval_count / total
        assert 0.1 < eval_ratio < 0.4, (
            f"Eval ratio {eval_ratio:.2f} is outside expected range for test_size=0.2"
        )

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
            training_config={"max_length": 2048, "packing": False},
        )
        train_packed, _ = create_ray_datasets(
            loader,
            tokenizer=tokenizer,
            training_config={"max_length": 2048, "packing": True},
        )
        unpacked_count = train_unpacked.count()
        packed_count = train_packed.count()
        assert packed_count < unpacked_count, (
            f"Packing should reduce row count by combining short samples, "
            f"but got packed={packed_count} >= unpacked={unpacked_count}"
        )


# === DPO tokenization ===


class TestTokenizationDPO:
    @pytest.fixture(scope="class")
    def ray_session(self):
        import ray

        if not ray.is_initialized():
            ray.init(address="local", num_cpus=2, runtime_env={"working_dir": None})
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

    def test_dpo_tokenization_content(self, dpo_datasets, tokenizer):
        train_ds, _ = dpo_datasets
        row = next(iter(train_ds.iter_rows()))
        assert "prompt_input_ids" in row
        assert "chosen_input_ids" in row
        assert "rejected_input_ids" in row

        # All sequences must be non-empty
        assert len(row["prompt_input_ids"]) > 0, "prompt_input_ids is empty"
        assert len(row["chosen_input_ids"]) > 0, "chosen_input_ids is empty"
        assert len(row["rejected_input_ids"]) > 0, "rejected_input_ids is empty"

        # Chosen and rejected must differ (otherwise DPO is meaningless)
        assert row["chosen_input_ids"] != row["rejected_input_ids"], (
            "chosen and rejected have identical token IDs"
        )

        # All token IDs must be within vocab range
        vocab_size = tokenizer.vocab_size
        for name in ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids"):
            for token_id in row[name]:
                assert 0 <= token_id < vocab_size, (
                    f"{name} has token ID {token_id} out of vocab range [0, {vocab_size})"
                )

    def test_dpo_eos_appended(self, dpo_datasets, tokenizer):
        train_ds, _ = dpo_datasets
        eos_id = tokenizer.eos_token_id
        # Check multiple rows, not just the first
        for i, row in enumerate(train_ds.iter_rows()):
            assert row["chosen_input_ids"][-1] == eos_id, (
                f"Row {i}: chosen missing EOS token"
            )
            assert row["rejected_input_ids"][-1] == eos_id, (
                f"Row {i}: rejected missing EOS token"
            )
            if i >= 9:
                break


# === Sharding correctness ===


class TestShardingCorrectness:
    @pytest.fixture(scope="class")
    def ray_session(self):
        import ray

        if not ray.is_initialized():
            ray.init(address="local", num_cpus=2, runtime_env={"working_dir": None})
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
        shards = tokenized_train_ds.split(2, equal=True)
        shard_0_ids = [tuple(row["input_ids"]) for row in shards[0].iter_rows()]
        shard_1_ids = [tuple(row["input_ids"]) for row in shards[1].iter_rows()]
        overlap = set(shard_0_ids) & set(shard_1_ids)
        assert len(overlap) == 0, f"Found {len(overlap)} duplicate rows across shards"

    def test_shard_sizes_roughly_equal(self, tokenized_train_ds):
        shards = tokenized_train_ds.split(2, equal=True)
        sizes = [shard.count() for shard in shards]
        assert abs(sizes[0] - sizes[1]) <= 1


# === Custom trainer DataLoader bypass ===


class TestCustomTrainerDataLoaders:
    def test_lfm_sft_trainer_no_distributed_sampler(self):
        from datasets import Dataset
        from torch.utils.data import DistributedSampler
        from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments

        from leap_finetune.training_loops.sft_run import LFMSFTTrainer

        dummy_dataset = Dataset.from_dict(
            {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        )
        args = TrainingArguments(
            output_dir="/tmp/test_sft",
            per_device_train_batch_size=1,
            report_to="none",
            use_cpu=True,
        )
        config = AutoConfig.from_pretrained("LiquidAI/LFM2-1.2B")
        model = AutoModelForCausalLM.from_config(config)
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
