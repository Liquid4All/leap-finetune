import torch
import torch.nn as nn

from leap_finetune.utils.moe_losses import (
    set_moe_sequence_partition_group,
    switch_load_balancing_loss,
)


def test_switch_load_balancing_loss_matches_non_cp_formula():
    router_probs = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    selected_experts = torch.tensor([[0], [1]])

    loss = switch_load_balancing_loss(
        router_probs=router_probs,
        selected_experts=selected_experts,
        num_experts=2,
        top_k=1,
        coeff=0.5,
    )

    aggregated_probs = router_probs.mean(dim=0)
    tokens_per_expert = torch.tensor([1.0, 1.0])
    expected = (aggregated_probs * tokens_per_expert).sum() * 2 * 0.5 / 2
    assert torch.allclose(loss, expected)


def test_switch_load_balancing_loss_uses_cp_global_counts_and_scales(monkeypatch):
    router_probs = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    selected_experts = torch.tensor([[0], [1]])
    group = object()

    def fake_all_reduce(tensor, op=None, group=None):
        tensor.add_(torch.tensor([1.0, 0.0]))

    monkeypatch.setattr(
        "leap_finetune.utils.moe_losses.dist.all_reduce",
        fake_all_reduce,
    )
    monkeypatch.setattr(
        "leap_finetune.utils.moe_losses.dist.get_world_size",
        lambda group=None: 2,
    )

    loss = switch_load_balancing_loss(
        router_probs=router_probs,
        selected_experts=selected_experts,
        num_experts=2,
        top_k=1,
        coeff=0.5,
        sequence_partition_group=group,
    )

    global_tokens_per_expert = torch.tensor([2.0, 1.0])
    local_prob_sums = router_probs.sum(dim=0)
    total_tokens = 4
    local_partial = (
        (local_prob_sums * global_tokens_per_expert).sum()
        * 2
        * 0.5
        / (total_tokens * total_tokens)
    )
    expected = local_partial * 2
    assert torch.allclose(loss, expected)


def test_set_moe_sequence_partition_group_updates_patched_blocks():
    class Lfm2MoeSparseMoeBlock(nn.Module):
        pass

    block = Lfm2MoeSparseMoeBlock()
    model = nn.Sequential(block)
    group = object()

    set_moe_sequence_partition_group(model, group)

    assert block._moe_sequence_partition_group is group
