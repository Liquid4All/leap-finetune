import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._functional_collectives import (
    all_to_all_single as _functional_a2a,
    all_to_all_single_autograd,
)

from leap_finetune.utils.moe_training import (
    InjectAuxLoss,
    switch_load_balancing_loss,
    z_loss,
)

logger = logging.getLogger(__name__)


# === Grouped MM Dispatch ===


def _grouped_mm(
    input_: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """Dispatch to torch.nn.functional.grouped_mm or torch._grouped_mm.

    Args:
        input_: [S, input_dim] — sorted tokens
        weight: [n_experts, input_dim, output_dim] — pre-transposed for A @ B
        offs: [n_experts] — cumulative token counts (int32)
    """
    input_ = input_.to(weight.dtype)
    if hasattr(torch.nn.functional, "grouped_mm"):
        return torch.nn.functional.grouped_mm(input_, weight, offs=offs)
    return torch._grouped_mm(input_, weight, offs=offs)


# === Token Permutation Utilities ===


def tokens_per_expert_from_routing(
    selected_experts: torch.Tensor, n_experts: int
) -> torch.Tensor:
    """Count tokens assigned to each expert via histogram (no Python loops).

    Args:
        selected_experts: [n_tokens, top_k] expert indices
        n_experts: total number of global experts

    Returns:
        tokens_per_expert: [n_experts]
    """
    flat = selected_experts.reshape(-1)
    return torch.histc(flat.float(), bins=n_experts, min=0, max=n_experts - 1).long()


def permute_tokens(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    n_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort tokens by expert assignment for AlltoAll dispatch.

    Each token-expert pair (from top-k) becomes one row in the output, grouped
    by expert index (0, 1, ..., n_experts-1).

    Args:
        hidden_states: [n_tokens, hidden_size]
        selected_experts: [n_tokens, top_k] expert indices per token
        routing_weights: [n_tokens, top_k] routing weight per assignment
        n_experts: total number of experts

    Returns:
        permuted_tokens: [n_permuted, hidden_size] sorted by expert
        permuted_weights: [n_permuted, 1] routing weight per assignment
        reverse_map: [n_permuted] original token indices for unpermutation
        tokens_per_expert: [n_experts] count of tokens assigned to each expert
    """
    n_tokens, top_k = selected_experts.shape
    device = hidden_states.device

    # Flatten top-k: each (token, slot) pair → one entry
    # token_idx[i] = which token, expert_ids[i] = which expert
    token_idx = (
        torch.arange(n_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )  # [n_tokens * top_k]
    expert_ids = selected_experts.reshape(-1)  # [n_tokens * top_k]
    flat_weights = routing_weights.reshape(-1)  # [n_tokens * top_k]

    # Sort by expert index — groups all tokens for expert 0 first, then 1, etc.
    perm = torch.argsort(expert_ids, stable=True)
    token_idx = token_idx[perm]
    flat_weights = flat_weights[perm]

    permuted_tokens = hidden_states[token_idx]
    permuted_weights = flat_weights.unsqueeze(-1)

    tokens_per_expert = tokens_per_expert_from_routing(selected_experts, n_experts)

    return permuted_tokens, permuted_weights, token_idx, tokens_per_expert


def unpermute_tokens(
    hidden_states: torch.Tensor,
    reverse_map: torch.Tensor,
    probs: torch.Tensor,
    n_tokens: int,
    hidden_size: int,
) -> torch.Tensor:
    """Scatter expert outputs back to original token order with weighted accumulation."""
    weighted = hidden_states * probs.to(hidden_states.dtype)
    output = torch.zeros(
        n_tokens, hidden_size, dtype=weighted.dtype, device=weighted.device
    )
    output.scatter_add_(0, reverse_map.unsqueeze(-1).expand_as(weighted), weighted)
    return output


# === EP Token Dispatcher ===


class EPTokenDispatcher:
    """AlltoAll dispatcher for Expert Parallelism.

    Token lifecycle:
    1. preprocess — exchange per-expert token counts via AlltoAll
    2. token_permutation — local sort by expert → AlltoAll → re-sort by local expert
    3. token_unpermutation — undo re-sort → reverse AlltoAll → unpermute
    """

    def __init__(
        self,
        n_local_experts: int,
        local_expert_indices: list[int],
        n_experts: int,
        top_k: int,
        ep_size: int,
        ep_group: dist.ProcessGroup,
    ):
        self.n_local_experts = n_local_experts
        self.local_expert_indices = local_expert_indices
        self.n_experts = n_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.ep_group = ep_group

        # Set during preprocess/permutation
        self.input_splits: list[int] = []
        self.output_splits: list[int] = []
        self.tokens_per_local_expert: torch.Tensor | None = None
        self._reverse_map: torch.Tensor | None = None
        self._permuted_weights: torch.Tensor | None = None
        self._unsort_indices: torch.Tensor | None = None
        self._n_tokens: int = 0
        self._hidden_size: int = 0

    def preprocess(self, tokens_per_expert: torch.Tensor) -> None:
        """Exchange per-expert token counts across EP ranks.

        Uses functional collectives (async, composable with FSDP2) instead of
        blocking dist.all_to_all_single which conflicts with FSDP2's scheduling.

        Args:
            tokens_per_expert: [n_experts] — counts from this rank's routing
        """
        # tokens_per_expert[e] = how many of MY tokens go to global expert e
        # Reshape to [ep_size, n_local_experts]: row r = counts for rank r's experts
        tpe_by_rank = tokens_per_expert.reshape(self.ep_size, self.n_local_experts)

        with torch.no_grad():
            # === 1. Exchange per-rank totals ===
            input_splits_t = tpe_by_rank.sum(dim=1).contiguous()
            output_splits_t = _functional_a2a(
                input_splits_t, None, None, group=self.ep_group
            )
            output_splits_t = torch.ops._c10d_functional.wait_tensor(output_splits_t)
            self.input_splits = input_splits_t.tolist()
            self.output_splits = output_splits_t.tolist()

            # === 2. Exchange per-expert counts ===
            expert_split = [self.n_local_experts] * self.ep_size
            tpe_flat = tpe_by_rank.flatten().contiguous()
            received_flat = _functional_a2a(
                tpe_flat, expert_split, expert_split, group=self.ep_group
            )
            received_flat = torch.ops._c10d_functional.wait_tensor(received_flat)

        # _received_tpe[r][e] = tokens from rank r for my local expert e
        self._received_tpe = received_flat.reshape(self.ep_size, self.n_local_experts)
        self.tokens_per_local_expert = self._received_tpe.sum(dim=0)

    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Sort tokens by expert → AlltoAll → re-sort by local expert.

        Args:
            hidden_states: [n_tokens, hidden_size]
            selected_experts: [n_tokens, top_k]
            routing_weights: [n_tokens, top_k]

        Returns:
            expert_input: [n_received, hidden_size] sorted by local expert
        """
        self._n_tokens = hidden_states.shape[0]
        self._hidden_size = hidden_states.shape[1]

        # === 1. Local permute: sort by global expert index ===
        permuted, weights, self._reverse_map, _ = permute_tokens(
            hidden_states, selected_experts, routing_weights, self.n_experts
        )
        self._permuted_weights = weights

        # === 2. AlltoAll: send tokens to owning EP rank ===
        # Uses functional collective (async, composable with FSDP2).
        global_tokens = all_to_all_single_autograd(
            permuted, self.output_splits, self.input_splits, self.ep_group
        )

        # === 3. Re-sort received tokens by local expert ===
        # After AlltoAll, tokens arrive as [from_rank0, from_rank1, ...].
        # Within each source chunk, tokens are sorted by local expert.
        # Re-sort so ALL tokens for expert 0 are contiguous, then expert 1, etc.
        # Always run resort even on empty tokens — branch divergence between DP
        # ranks causes NCCL operation ordering drift and deadlock at dp_size=2.
        global_tokens, self._unsort_indices = self._resort_by_expert(global_tokens)

        return global_tokens

    def _resort_by_expert(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-sort received tokens so each local expert's tokens are contiguous.

        Builds expert-id labels from _received_tpe using repeat_interleave
        (vectorized, no Python loops), then argsorts to group by expert.
        """
        # _received_tpe: [ep_size, n_local_experts] — counts per (source_rank, expert)
        # Flatten to [ep_size * n_local_experts], with expert ids cycling 0..E-1
        device = tokens.device
        counts = self._received_tpe.flatten()  # [ep_size * n_local_experts]
        expert_ids = torch.arange(
            self.n_local_experts, device=device
        ).repeat(self.ep_size)  # [0,1,..,E-1, 0,1,..,E-1, ...]
        expert_labels = expert_ids.repeat_interleave(counts)

        resort_indices = expert_labels.argsort(stable=True)
        unsort_indices = torch.empty_like(resort_indices)
        unsort_indices[resort_indices] = torch.arange(
            len(resort_indices), device=device
        )

        return tokens[resort_indices], unsort_indices

    def token_unpermutation(self, expert_output: torch.Tensor) -> torch.Tensor:
        """Undo re-sort → reverse AlltoAll → unpermute with routing weights.

        Args:
            expert_output: [n_received, hidden_size] from local expert compute

        Returns:
            output: [n_tokens, hidden_size] in original token order
        """
        # === 1. Undo expert re-sort (back to source-rank order) ===
        # Always run unsort to avoid branch divergence between DP ranks.
        if self._unsort_indices is not None:
            expert_output = expert_output[self._unsort_indices]

        # === 2. Reverse AlltoAll: send outputs back to originating ranks ===
        local_tokens = all_to_all_single_autograd(
            expert_output, self.input_splits, self.output_splits, self.ep_group
        )

        # === 3. Unpermute to original token order with routing weights ===
        output = unpermute_tokens(
            local_tokens,
            self._reverse_map,
            self._permuted_weights,
            self._n_tokens,
            self._hidden_size,
        )
        return output


# === Local Expert Compute ===


def compute_local_experts(
    experts: nn.Module,
    tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Run tokens through local experts using grouped_mm.

    Uses torch grouped_mm to batch all local experts into two kernel launches
    (gate_up + down) instead of looping per expert.

    Args:
        experts: Lfm2MoeExperts with sharded weights
            gate_up_proj: [n_local, 2*intermediate, hidden]
            down_proj: [n_local, hidden, intermediate]
        tokens: [n_tokens, hidden_size] sorted by expert
        tokens_per_expert: [n_local_experts]
    """
    # Always run grouped_mm — branch divergence between DP ranks causes
    # NCCL operation ordering drift and deadlock at dp_size=2.
    # Offsets: cumulative token counts per expert — [n_local_experts] int32
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    # gate_up_proj: [n_local, 2*intermediate, hidden] → transpose to [n_local, hidden, 2*intermediate]
    # grouped_mm: [S, hidden] @ [n_local, hidden, 2*intermediate] → [S, 2*intermediate]
    gate_up_out = _grouped_mm(
        tokens, experts.gate_up_proj.transpose(-2, -1), offsets
    )
    gate, up = gate_up_out.chunk(2, dim=-1)
    activated = experts.act_fn(gate) * up

    # down_proj: [n_local, hidden, intermediate] → transpose to [n_local, intermediate, hidden]
    # grouped_mm: [S, intermediate] @ [n_local, intermediate, hidden] → [S, hidden]
    return _grouped_mm(activated, experts.down_proj.transpose(-2, -1), offsets)


# === EP-Enhanced MoE Forward Patch ===


def patch_moe_block_for_ep(
    block: nn.Module,
    dispatcher: EPTokenDispatcher,
    moe_config=None,
) -> None:
    """Monkey-patch an Lfm2MoeSparseMoeBlock for EP-aware forward.

    Integrates aux loss computation (load balancing, z-loss) directly so that
    applying EP after MoETrainingEnhancer doesn't lose the losses.

    Args:
        block: Lfm2MoeSparseMoeBlock
        dispatcher: EPTokenDispatcher
        moe_config: MoETrainingConfig (optional, for aux losses)
    """
    experts = block.experts

    def ep_moe_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        x = hidden_states.view(-1, H)  # [n_tokens, H]

        # === 1. Router ===
        router_logits = block.gate(x)  # [n_tokens, E]

        n_experts = router_logits.shape[-1]

        # === 2. Inject aux losses into intermediates (BEFORE expert dispatch) ===
        # Losses are injected into router_logits/routing_weights so their backward
        # flows through the same layer pass as the main gradient. Injecting into
        # the final output would trigger extra FSDP all-gathers after the layer's
        # reduce-scatter, desyncing DP ranks.
        if moe_config is not None:
            if moe_config.z_loss_coef > 0:
                zl = z_loss(router_logits, moe_config.z_loss_coef)
                router_logits = InjectAuxLoss.apply(router_logits, zl)
                block._moe_z_loss = zl.detach()

        selected_experts, routing_weights = block.route_tokens_to_experts(
            router_logits
        )  # [n_tokens, K], [n_tokens, K]
        top_k = selected_experts.shape[-1]

        if moe_config is not None:
            if moe_config.aux_loss_coef > 0:
                router_probs = torch.sigmoid(router_logits)
                aux = switch_load_balancing_loss(
                    router_probs,
                    selected_experts,
                    n_experts,
                    top_k,
                    moe_config.aux_loss_coef,
                )
                routing_weights = InjectAuxLoss.apply(routing_weights, aux)
                block._moe_aux_loss = aux.detach()

        # === 3. EP dispatch: permute → AlltoAll → re-sort ===
        tpe = tokens_per_expert_from_routing(selected_experts, n_experts)
        dispatcher.preprocess(tpe)
        global_tokens = dispatcher.token_permutation(
            x, selected_experts, routing_weights
        )

        # === 4. Local expert compute (grouped_mm) ===
        # Always run — branch divergence causes NCCL ordering drift at dp_size=2.
        expert_output = compute_local_experts(
            experts, global_tokens, dispatcher.tokens_per_local_expert
        )

        # === 5. Gather outputs: unsort → AlltoAll → unpermute ===
        output = dispatcher.token_unpermutation(expert_output)
        output = output.view(B, S, H)

        # === 6. Metrics ===
        if moe_config is not None:
            with torch.no_grad():
                tokens_per_expert = torch.zeros(
                    n_experts, device=x.device, dtype=torch.float32
                )
                ones = torch.ones_like(selected_experts[:, 0], dtype=torch.float32)
                for k in range(top_k):
                    tokens_per_expert.scatter_add_(0, selected_experts[:, k], ones)
                block._moe_tokens_per_expert = tokens_per_expert
                block._moe_router_logits_mean = router_logits.mean().item()
                block._moe_router_logits_std = router_logits.std().item()
                block._moe_num_experts = n_experts

        return output

    block.forward = ep_moe_forward


def apply_ep_to_model(model: nn.Module, ep_config: dict, moe_config=None) -> None:
    """Apply EP dispatching to all MoE blocks in the model.

    Must be called after shard_experts(). If moe_config is provided, aux losses
    are computed in the EP forward (replaces MoETrainingEnhancer).

    Args:
        model: model with sharded experts
        ep_config: dict from create_ep_mesh()
        moe_config: MoETrainingConfig for aux loss computation
    """
    patched = 0

    for module in model.modules():
        if type(module).__name__ != "Lfm2MoeSparseMoeBlock":
            continue

        top_k = getattr(module, "top_k", 2)

        dispatcher = EPTokenDispatcher(
            n_local_experts=ep_config["n_local_experts"],
            local_expert_indices=ep_config["local_expert_indices"],
            n_experts=ep_config["num_experts"],
            top_k=top_k,
            ep_size=ep_config["ep_size"],
            ep_group=ep_config["ep_group"],
        )

        patch_moe_block_for_ep(module, dispatcher, moe_config=moe_config)
        patched += 1

    logger.info(f"Applied EP dispatch to {patched} MoE blocks")
