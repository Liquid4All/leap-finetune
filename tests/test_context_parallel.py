from unittest.mock import patch

import torch

from leap_finetune.utils.context_parallel import split_batch_for_cp, validate_cp_config


def test_split_batch_for_cp_uses_load_balanced_chunks():
    batch = {
        "input_ids": torch.arange(8, dtype=torch.long).unsqueeze(0),
        "labels": torch.arange(100, 108, dtype=torch.long).unsqueeze(0),
    }

    rank0 = split_batch_for_cp(batch, cp_rank=0, cp_size=2)
    rank1 = split_batch_for_cp(batch, cp_rank=1, cp_size=2)

    assert torch.equal(rank0["input_ids"], torch.tensor([[0, 1, 6, 7]]))
    assert torch.equal(rank1["input_ids"], torch.tensor([[2, 3, 4, 5]]))
    assert torch.equal(rank0["labels"], torch.tensor([[100, 101, 106, 107]]))
    assert torch.equal(rank1["labels"], torch.tensor([[102, 103, 104, 105]]))
    assert torch.equal(rank0["position_ids"], torch.tensor([[0, 1, 6, 7]]))
    assert torch.equal(rank1["position_ids"], torch.tensor([[2, 3, 4, 5]]))


def test_split_batch_for_cp_pads_before_chunk_selection():
    batch = {
        "input_ids": torch.arange(7, dtype=torch.long).unsqueeze(0),
        "labels": torch.arange(10, 17, dtype=torch.long).unsqueeze(0),
    }

    rank0 = split_batch_for_cp(batch, cp_rank=0, cp_size=2)
    rank1 = split_batch_for_cp(batch, cp_rank=1, cp_size=2)

    assert torch.equal(rank0["input_ids"], torch.tensor([[0, 1, 6, 0]]))
    assert torch.equal(rank1["input_ids"], torch.tensor([[2, 3, 4, 5]]))
    assert torch.equal(rank0["labels"], torch.tensor([[10, 11, 16, -100]]))
    assert torch.equal(rank1["labels"], torch.tensor([[12, 13, 14, 15]]))
    assert torch.equal(rank0["position_ids"], torch.tensor([[0, 1, 6, 7]]))
    assert torch.equal(rank1["position_ids"], torch.tensor([[2, 3, 4, 5]]))


def test_validate_cp_config_allows_non_divisible_max_length_with_runtime_padding():
    with patch("transformers.utils.is_flash_attn_2_available", return_value=True):
        validate_cp_config(cp_size=2, max_length=7, world_size=4)
