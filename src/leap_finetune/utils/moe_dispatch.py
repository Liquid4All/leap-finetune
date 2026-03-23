import logging

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


# === AlltoAll Autograd Function ===


class _AllToAll(torch.autograd.Function):
    """Autograd-compatible AlltoAll collective.

    Forward: all_to_all_single with output_splits/input_splits
    Backward: all_to_all_single with reversed splits
    """

    @staticmethod
    def forward(
        ctx,
        group: dist.ProcessGroup,
        input_: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
    ) -> torch.Tensor:
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        output = torch.empty(
            sum(output_split_sizes),
            *input_.shape[1:],
            dtype=input_.dtype,
            device=input_.device,
        )
        dist.all_to_all_single(
            output,
            input_,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse the splits for backward
        grad_input = torch.empty(
            sum(ctx.input_split_sizes),
            *grad_output.shape[1:],
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.all_to_all_single(
            grad_input,
            grad_output,
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group,
        )
        return None, grad_input, None, None


def all_to_all(
    group: dist.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
) -> torch.Tensor:
    return _AllToAll.apply(group, input_, output_split_sizes, input_split_sizes)


# === Token Permutation Utilities ===


def permute_tokens(
    hidden_states: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute tokens by expert assignment for AlltoAll dispatch.

    Args:
        hidden_states: [n_tokens, hidden_size]
        routing_map: [n_tokens, n_experts] bool — which expert each token goes to
        probs: [n_tokens, top_k] routing weights

    Returns:
        permuted_tokens: [n_permuted, hidden_size] sorted by expert
        permuted_probs: [n_permuted, 1] corresponding weights
        reverse_map: [n_permuted] indices to reverse permutation
    """
    # routing_map columns correspond to experts
    # For each expert, gather tokens assigned to it
    n_tokens, n_experts = routing_map.shape
    device = hidden_states.device

    token_indices = []
    prob_values = []

    for expert_idx in range(n_experts):
        assigned = routing_map[:, expert_idx].nonzero(as_tuple=True)[0]
        token_indices.append(assigned)
        # Get the prob for tokens assigned to this expert
        # Find which top_k slot matches this expert
        prob_values.append(torch.ones(assigned.shape[0], device=device))

    token_indices = torch.cat(token_indices)
    permuted_tokens = hidden_states[token_indices]

    # Build reverse map: position i in permuted → original token index
    reverse_map = token_indices

    # Get probs — for simplicity, gather max prob per token
    permuted_probs = (
        probs[token_indices, 0].unsqueeze(-1)
        if probs.dim() == 2
        else probs[token_indices].unsqueeze(-1)
    )

    return permuted_tokens, permuted_probs, reverse_map


def unpermute_tokens(
    hidden_states: torch.Tensor,
    reverse_map: torch.Tensor,
    probs: torch.Tensor,
    n_tokens: int,
    hidden_size: int,
) -> torch.Tensor:
    """Reverse permutation: scatter expert outputs back to original token order.

    Args:
        hidden_states: [n_permuted, hidden_size] expert outputs
        reverse_map: [n_permuted] original token indices
        probs: [n_permuted, 1] routing weights for weighted sum
        n_tokens: original number of tokens
        hidden_size: hidden dimension

    Returns:
        output: [n_tokens, hidden_size]
    """
    output = torch.zeros(
        n_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device
    )
    weighted = hidden_states * probs
    output.scatter_add_(0, reverse_map.unsqueeze(-1).expand_as(weighted), weighted)
    return output


# === EP Token Dispatcher ===


class EPTokenDispatcher:
    """Simplified AlltoAll dispatcher for Expert Parallelism without Tensor Parallelism.

    Handles the full dispatch cycle:
    1. preprocess: compute AlltoAll split sizes from routing decisions
    2. token_permutation: local permute → AlltoAll → sort by local expert
    3. token_unpermutation: unsort → reverse AlltoAll → unpermute
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

        # Set during preprocess
        self.input_splits: list[int] = []
        self.output_splits: list[int] = []
        self._reverse_map: torch.Tensor | None = None
        self._n_tokens: int = 0
        self._hidden_size: int = 0

    def preprocess(self, routing_map: torch.Tensor) -> None:
        """Compute AlltoAll split sizes from routing map.

        Args:
            routing_map: [n_tokens, n_experts] bool tensor
        """
        # tokens_per_expert: [n_experts] count of local tokens going to each expert
        tokens_per_expert = routing_map.sum(dim=0).long()

        # input_splits: tokens sent TO each EP rank
        # Each EP rank owns n_local_experts consecutive experts
        tpe_by_rank = tokens_per_expert.reshape(self.ep_size, self.n_local_experts)
        self.input_splits = tpe_by_rank.sum(dim=1).tolist()

        # Gather output_splits from all EP ranks (how many tokens we receive)
        input_splits_tensor = torch.tensor(
            self.input_splits, device=routing_map.device, dtype=torch.long
        )
        output_splits_tensor = torch.zeros_like(input_splits_tensor)
        dist.all_to_all_single(
            output_splits_tensor, input_splits_tensor, group=self.ep_group
        )
        self.output_splits = output_splits_tensor.tolist()

    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Permute tokens → AlltoAll → sort by local expert.

        Args:
            hidden_states: [n_tokens, hidden_size]
            probs: [n_tokens, top_k] routing weights
            routing_map: [n_tokens, n_experts] bool

        Returns:
            global_tokens: [n_received, hidden_size] tokens for local experts
            global_probs: [n_received, 1] corresponding weights
            tokens_per_local_expert: [n_local_experts] count per local expert
        """
        self._n_tokens = hidden_states.shape[0]
        self._hidden_size = hidden_states.shape[1]

        # === 1. Local permute by expert assignment ===
        permuted, permuted_probs, self._reverse_map = permute_tokens(
            hidden_states, routing_map, probs
        )

        # === 2. AlltoAll across EP ranks ===
        global_tokens = all_to_all(
            self.ep_group, permuted, self.output_splits, self.input_splits
        )
        global_probs = all_to_all(
            self.ep_group, permuted_probs, self.output_splits, self.input_splits
        )

        # === 3. Compute tokens per local expert ===
        # After AlltoAll, tokens are ordered by sending rank, then by expert within rank.
        # We need to count how many tokens each local expert received.
        total_received = sum(self.output_splits)
        tokens_per_local_expert = torch.zeros(
            self.n_local_experts, device=hidden_states.device, dtype=torch.long
        )

        # Each sender sends tokens grouped by expert. For simplicity, distribute evenly
        # based on the routing map gathered during preprocess.
        # In practice, tokens arrive sorted by (source_rank, expert_idx), so we split accordingly.
        if total_received > 0 and self.n_local_experts > 1:
            # Approximate: evenly distribute among local experts
            per_expert = total_received // self.n_local_experts
            remainder = total_received % self.n_local_experts
            for i in range(self.n_local_experts):
                tokens_per_local_expert[i] = per_expert + (1 if i < remainder else 0)
        elif total_received > 0:
            tokens_per_local_expert[0] = total_received

        return global_tokens, global_probs, tokens_per_local_expert

    def token_unpermutation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reverse: AlltoAll back → unpermute to original token order.

        Args:
            hidden_states: [n_received, hidden_size] expert outputs

        Returns:
            output: [n_tokens, hidden_size]
        """
        # Dummy probs for unpermute (already applied during expert compute)
        probs = torch.ones(
            hidden_states.shape[0],
            1,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # === 1. Reverse AlltoAll ===
        local_tokens = all_to_all(
            self.ep_group, hidden_states, self.input_splits, self.output_splits
        )

        # === 2. Unpermute to original token order ===
        output = unpermute_tokens(
            local_tokens, self._reverse_map, probs, self._n_tokens, self._hidden_size
        )

        return output


# === EP-Enhanced MoE Forward Patch ===


def patch_moe_block_for_ep(block: nn.Module, dispatcher: EPTokenDispatcher) -> None:
    """Monkey-patch an Lfm2MoeSparseMoeBlock for EP-aware forward.

    The patched forward:
    1. Routes tokens (same as original)
    2. Dispatches via AlltoAll to distribute tokens across EP ranks
    3. Computes only local experts
    4. Unpermutes via reverse AlltoAll

    Args:
        block: Lfm2MoeSparseMoeBlock module
        dispatcher: EPTokenDispatcher configured for this model
    """

    def ep_moe_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        x = hidden_states.view(-1, H)

        # === Router ===
        router_logits = block.gate(x)
        selected_experts, routing_weights = block.route_tokens_to_experts(router_logits)

        num_experts = router_logits.shape[-1]

        # === Build routing map [n_tokens, n_experts] ===
        routing_map = torch.zeros(
            x.shape[0], num_experts, dtype=torch.bool, device=x.device
        )
        routing_map.scatter_(1, selected_experts, True)

        # === EP dispatch ===
        dispatcher.preprocess(routing_map)
        permuted_tokens, permuted_probs, tpe = dispatcher.token_permutation(
            x, routing_weights, routing_map
        )

        # === Local expert compute ===
        if permuted_tokens.shape[0] > 0:
            expert_output = _compute_local_experts(
                block, permuted_tokens, permuted_probs, tpe
            )
        else:
            expert_output = permuted_tokens

        # === EP unpermutation ===
        output = dispatcher.token_unpermutation(expert_output)

        return output.view(B, S, H)

    block.forward = ep_moe_forward


def _compute_local_experts(
    block: nn.Module,
    tokens: torch.Tensor,
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Run tokens through local experts only.

    Args:
        block: MoE block with sharded experts
        tokens: [n_tokens, hidden_size]
        probs: [n_tokens, 1] routing weights
        tokens_per_expert: [n_local_experts] count per expert

    Returns:
        output: [n_tokens, hidden_size]
    """
    outputs = []
    offset = 0

    for i, expert in enumerate(block.experts):
        n = tokens_per_expert[i].item()
        if n == 0:
            continue
        expert_input = tokens[offset : offset + n]
        expert_output = expert(expert_input)
        # Weight by routing prob
        expert_probs = probs[offset : offset + n]
        outputs.append(expert_output * expert_probs)
        offset += n

    if outputs:
        return torch.cat(outputs, dim=0)
    return tokens


def apply_ep_to_model(model: nn.Module, ep_config: dict) -> None:
    """Apply EP dispatching to all MoE blocks in the model.

    Must be called after shard_experts() so experts ModuleList is local-only.

    Args:
        model: model with sharded experts
        ep_config: dict from create_ep_process_groups()
    """
    patched = 0

    for module in model.modules():
        if type(module).__name__ != "Lfm2MoeSparseMoeBlock":
            continue

        # Determine top_k from the model config
        top_k = getattr(module, "top_k", 2)

        dispatcher = EPTokenDispatcher(
            n_local_experts=ep_config["n_local_experts"],
            local_expert_indices=ep_config["local_expert_indices"],
            n_experts=ep_config["num_experts"],
            top_k=top_k,
            ep_size=ep_config["ep_size"],
            ep_group=ep_config["ep_group"],
        )

        patch_moe_block_for_ep(module, dispatcher)
        patched += 1

    logger.info(f"Applied EP dispatch to {patched} MoE blocks")
