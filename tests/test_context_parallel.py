import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from leap_finetune.utils.context_parallel import (
    _prepend_cp_left_halo,
    compute_cp_causal_lm_loss,
    prefix_gather_attention,
    split_batch_for_cp,
    validate_cp_batch_replicated,
    validate_cp_config,
    validate_cp_model_support,
)

def test_split_batch_for_cp_uses_consecutive_chunks():
    batch = {
        "input_ids": torch.arange(8, dtype=torch.long).unsqueeze(0),
        "labels": torch.arange(100, 108, dtype=torch.long).unsqueeze(0),
    }

    rank0 = split_batch_for_cp(batch, cp_rank=0, cp_size=2)
    rank1 = split_batch_for_cp(batch, cp_rank=1, cp_size=2)

    assert torch.equal(rank0["input_ids"], torch.tensor([[0, 1, 2, 3]]))
    assert torch.equal(rank1["input_ids"], torch.tensor([[4, 5, 6, 7]]))
    assert torch.equal(rank0["labels"], torch.tensor([[100, 101, 102, 103]]))
    assert torch.equal(rank1["labels"], torch.tensor([[104, 105, 106, 107]]))
    assert torch.equal(rank0["shift_labels"], torch.tensor([[101, 102, 103, 104]]))
    assert torch.equal(rank1["shift_labels"], torch.tensor([[105, 106, 107, -100]]))
    assert torch.equal(rank0["position_ids"], torch.tensor([[0, 1, 2, 3]]))
    assert torch.equal(rank1["position_ids"], torch.tensor([[4, 5, 6, 7]]))


def test_split_batch_for_cp_pads_before_consecutive_chunk_selection():
    batch = {
        "input_ids": torch.arange(7, dtype=torch.long).unsqueeze(0),
        "labels": torch.arange(10, 17, dtype=torch.long).unsqueeze(0),
    }

    rank0 = split_batch_for_cp(batch, cp_rank=0, cp_size=2)
    rank1 = split_batch_for_cp(batch, cp_rank=1, cp_size=2)

    assert torch.equal(rank0["input_ids"], torch.tensor([[0, 1, 2, 3]]))
    assert torch.equal(rank1["input_ids"], torch.tensor([[4, 5, 6, 0]]))
    assert torch.equal(rank0["labels"], torch.tensor([[10, 11, 12, 13]]))
    assert torch.equal(rank1["labels"], torch.tensor([[14, 15, 16, -100]]))
    assert torch.equal(rank0["shift_labels"], torch.tensor([[11, 12, 13, 14]]))
    assert torch.equal(rank1["shift_labels"], torch.tensor([[15, 16, -100, -100]]))
    assert torch.equal(rank0["position_ids"], torch.tensor([[0, 1, 2, 3]]))
    assert torch.equal(rank1["position_ids"], torch.tensor([[4, 5, 6, 7]]))


def test_split_batch_for_cp_preserves_local_padding_mask_copy():
    batch = {
        "input_ids": torch.arange(8, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long),
    }

    rank1 = split_batch_for_cp(batch, cp_rank=1, cp_size=2)

    expected = torch.tensor([[1, 0, 0, 0]], dtype=torch.long)
    assert torch.equal(rank1["attention_mask"], expected)
    assert torch.equal(rank1["leap_cp_padding_mask"], expected)


def test_validate_cp_config_allows_non_divisible_max_length_with_runtime_padding():
    with patch("transformers.utils.is_flash_attn_2_available", return_value=True):
        validate_cp_config(cp_size=2, max_length=7, world_size=4)


class _DummyHybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"layer_types": ["conv", "full_attention"]})()


def test_validate_cp_model_support_rejects_packed_hybrid_conv():
    model = _DummyHybridModel()
    with patch("transformers.utils.is_flash_attn_2_available", return_value=True):
        with pytest.raises(ValueError, match="packing=True is not supported"):
            validate_cp_model_support(
                model,
                {"context_parallel_size": 2, "packing": True},
            )


def test_prefix_gather_attention_uses_contiguous_visible_prefix(monkeypatch):
    captured = {}

    def fake_flash_attn_func(q, k, v, causal=True):
        captured["q_shape"] = q.shape
        captured["k_shape"] = k.shape
        captured["v_shape"] = v.shape
        captured["causal"] = causal
        return q

    monkeypatch.setitem(
        sys.modules,
        "flash_attn",
        SimpleNamespace(flash_attn_func=fake_flash_attn_func),
    )
    monkeypatch.setattr(
        "leap_finetune.utils.context_parallel._all_gather_fixed_seq",
        lambda tensor, group: [
            torch.full_like(tensor, fill_value=rank + 1) for rank in range(4)
        ],
    )

    q = torch.zeros(1, 3, 2, 4)
    k = torch.zeros(1, 3, 2, 4)
    v = torch.zeros(1, 3, 2, 4)
    out = prefix_gather_attention(q, k, v, cp_group="group", cp_rank=1, cp_size=4)

    assert torch.equal(out, q)
    assert captured["causal"] is True
    assert captured["q_shape"] == (1, 3, 2, 4)
    # rank 1 should see rank 0 + rank 1 = 2 contiguous chunks
    assert captured["k_shape"] == (1, 6, 2, 4)
    assert captured["v_shape"] == (1, 6, 2, 4)


def test_prepend_cp_left_halo_uses_previous_rank_tail():
    hidden_states = torch.tensor([[[10.0], [11.0], [12.0]]])
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

    with patch(
        "leap_finetune.utils.context_parallel._all_gather_cp_halos",
        side_effect=[
            [
                torch.tensor([[[1.0], [2.0]]]),
                torch.tensor([[[3.0], [4.0]]]),
                torch.tensor([[[5.0], [6.0]]]),
            ],
            [
                torch.tensor([[1, 1]], dtype=torch.long),
                torch.tensor([[1, 0]], dtype=torch.long),
                torch.tensor([[0, 0]], dtype=torch.long),
            ],
        ],
    ):
        extended_hidden, extended_mask = _prepend_cp_left_halo(
            hidden_states,
            attention_mask,
            cp_group="group",
            cp_rank=2,
            halo_width=2,
        )

    assert torch.equal(
        extended_hidden,
        torch.tensor([[[3.0], [4.0], [10.0], [11.0], [12.0]]]),
    )
    assert torch.equal(
        extended_mask,
        torch.tensor([[1, 0, 1, 1, 1]], dtype=torch.long),
    )


class _DummyCausalLM(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits)

    def forward(self, input_ids):
        return {"logits": self.logits.expand(input_ids.shape[0], -1, -1)}


def test_cp_causal_lm_loss_scales_before_distributed_gradient_average(monkeypatch):
    def fake_all_reduce(tensor, op=None, group=None):
        tensor.mul_(2)

    monkeypatch.setattr(
        "leap_finetune.utils.context_parallel.dist.all_reduce",
        fake_all_reduce,
    )

    logits = torch.zeros(1, 3, 5)
    model = _DummyCausalLM(logits)
    inputs = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "labels": torch.tensor([[-100, 1, -100]], dtype=torch.long),
    }

    loss = compute_cp_causal_lm_loss(
        model,
        inputs,
        cp_group="group",
        cp_size=4,
    )

    local_token_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        inputs["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(loss, local_token_loss * 2)

    loss.backward()
    assert model.logits.grad is not None


def test_cp_causal_lm_loss_uses_precomputed_shift_labels(monkeypatch):
    def fake_all_reduce(tensor, op=None, group=None):
        return None

    monkeypatch.setattr(
        "leap_finetune.utils.context_parallel.dist.all_reduce",
        fake_all_reduce,
    )

    logits = torch.zeros(1, 2, 5)
    logits[0, 0, 4] = 10.0
    logits[0, 1, 3] = 10.0
    model = _DummyCausalLM(logits)
    inputs = {
        "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "labels": torch.tensor([[0, 1]], dtype=torch.long),
        "shift_labels": torch.tensor([[4, -100]], dtype=torch.long),
    }

    loss = compute_cp_causal_lm_loss(
        model,
        inputs,
        cp_group="group",
        cp_size=1,
    )

    assert loss.item() < 0.001


def test_validate_cp_batch_replicated_rejects_mismatched_inputs(monkeypatch):
    def fake_all_gather_object(output, fingerprint, group=None):
        output[0] = fingerprint
        output[1] = {"input_ids": ((1, 3), 999, (9, 9, 9), (9, 9, 9))}

    monkeypatch.setattr(
        "leap_finetune.utils.context_parallel.dist.all_gather_object",
        fake_all_gather_object,
    )

    with pytest.raises(RuntimeError, match="different pre-split batches"):
        validate_cp_batch_replicated(
            {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)},
            cp_group="group",
            cp_rank=0,
            cp_size=2,
        )
